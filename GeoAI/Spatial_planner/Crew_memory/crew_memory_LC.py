#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np


# In[2]:


from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate


# In[3]:


class PtExampleSelector(BaseExampleSelector):
    """
    LangChain-compatible ExampleSelector that uses a precomputed torch tensor of embeddings (.pt),
    or computes them with HuggingFaceEmbeddings if absent.
    Each example must contain an "instruction" field used as the searchable text.
    """

    def __init__(
        self,
        examples: List[Dict],
        embeddings_file: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        instruction_key: str = "instruction",
        device: Optional[str] = None,  # "cpu" | "cuda" | None (auto)
    ):
        self.examples = examples
        self.instruction_key = instruction_key
        self.instructions = [ex.get(self.instruction_key, "") for ex in self.examples]

        # Embedding model (LangChain wrapper around sentence-transformers)
        self.embed = HuggingFaceEmbeddings(model_name=model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.embeddings_path = Path(embeddings_file) if embeddings_file else None
        self.emb_t: Optional[torch.Tensor] = None  # (N, d)

        self._load_or_build_embeddings()

    def _load_or_build_embeddings(self):
        # Try loading .pt
        if self.embeddings_path and self.embeddings_path.exists():
            self.emb_t = torch.load(self.embeddings_path, map_location=self.device)
            if not isinstance(self.emb_t, torch.Tensor):
                self.emb_t = torch.tensor(self.emb_t, device=self.device)
        else:
            # Compute via LangChain wrapper (returns list[list[float]])
            if len(self.instructions) == 0:
                self.emb_t = torch.empty((0, 0), device=self.device)
                return
            vecs = self.embed.embed_documents(self.instructions)  # List[List[float]]
            self.emb_t = torch.tensor(np.array(vecs), device=self.device, dtype=torch.float32)
            if self.embeddings_path:
                torch.save(self.emb_t, self.embeddings_path)

        # Normalize once for cosine similarity
        if self.emb_t.numel() > 0:
            self.emb_t = F.normalize(self.emb_t, dim=1)

    # ---- BaseExampleSelector API ----
    def add_example(self, example: Dict) -> None:
        """Append a new example and update embeddings on the fly."""
        self.examples.append(example)
        text = example.get(self.instruction_key, "")
        if not text:
            # still keep it, but skip embedding
            return
        v = self.embed.embed_query(text)
        v_t = torch.tensor(np.array(v), device=self.device, dtype=torch.float32).unsqueeze(0)
        v_t = F.normalize(v_t, dim=1)

        if self.emb_t is None or self.emb_t.numel() == 0:
            self.emb_t = v_t
        else:
            self.emb_t = torch.cat([self.emb_t, v_t], dim=0)

        self.instructions.append(text)
        # Persist if path was provided
        if self.embeddings_path:
            torch.save(self.emb_t, self.embeddings_path)

    def select_examples(self, input_variables: Dict) -> List[Dict]:
        """
        LangChain calls this with your input variables.
        By convention we’ll look for 'user_prompt' first, else any single string value.
        """
        if not self.examples:
            return []

        query = input_variables.get("user_prompt")
        if query is None:
            # Pick the first str in the dict as query (fallback)
            for v in input_variables.values():
                if isinstance(v, str) and v.strip():
                    query = v
                    break
        if not query:
            # No query -> return first few examples
            return self.examples[: min(3, len(self.examples))]

        # Embed query and cosine-sim against stored embeddings
        q = self.embed.embed_query(query)
        q_t = torch.tensor(np.array(q), device=self.device, dtype=torch.float32).unsqueeze(0)
        q_t = F.normalize(q_t, dim=1)

        # Shapes: q_t = (1, d), emb_t = (N, d)
        sims = (q_t @ self.emb_t.T).squeeze(0)  # cosine similarity after normalization
        k = min(3, sims.shape[0])  # default top-3 for selector
        topk = torch.topk(sims, k=k).indices.tolist()
        return [self.examples[i] for i in topk]

    # ---- Convenience methods (not required by BaseExampleSelector) ----
    def top_k(self, query: str, k: int = 3) -> List[Dict]:
        if not self.examples:
            return []
        q = self.embed.embed_query(query)
        q_t = torch.tensor(np.array(q), device=self.device, dtype=torch.float32).unsqueeze(0)
        q_t = F.normalize(q_t, dim=1)
        sims = (q_t @ self.emb_t.T).squeeze(0)
        k = min(k, sims.shape[0])
        idxs = torch.topk(sims, k=k).indices.tolist()
        return [self.examples[i] for i in idxs]


# In[4]:


class CrewMemoryLC:
    """
    High-level wrapper so you can keep calling .top_k() / .get_similar_examples()
    and also get a LangChain-compatible FewShotPromptTemplate.
    """

    def __init__(
        self,
        memory_file: str,
        embeddings_file: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        instruction_key: str = "instruction",
        device: Optional[str] = None,
    ):
        self.memory_path = Path(memory_file)
        if not self.memory_path.exists():
            raise FileNotFoundError(f"Memory file not found: {self.memory_path}")

        self.examples: List[Dict] = json.loads(self.memory_path.read_text(encoding="utf-8"))

        self.selector = PtExampleSelector(
            examples=self.examples,
            embeddings_file=embeddings_file,
            model_name=model_name,
            instruction_key=instruction_key,
            device=device,
        )

    # ---- Backwards-friendly API ----
    def get_similar_examples(self, new_instruction: str, top_k: int = 2) -> List[Dict]:
        return self.selector.top_k(new_instruction, k=top_k)

    def top_k(self, query: str, k: int = 3) -> List[Dict]:
        return self.selector.top_k(query, k=k)

    def add_example(self, example: Dict) -> None:
        """Append to JSON + update embedding + persist JSON."""
        self.selector.add_example(example)
        self.examples.append(example)
        self.memory_path.write_text(json.dumps(self.examples, indent=2), encoding="utf-8")

    # ---- LangChain FewShot integration ----
    def make_fewshot_prompt(
        self,
        example_prompt: PromptTemplate,
        suffix: str,
        input_variables: List[str],
        k_default: int = 3,
    ) -> FewShotPromptTemplate:
        """
        Returns a FewShotPromptTemplate that calls our selector.
        LangChain will use select_examples() under the hood.
        """
        # Note: k_default is controlled inside select_examples (top-3). You can change that by editing PtExampleSelector.
        return FewShotPromptTemplate(
            example_selector=self.selector,
            example_prompt=example_prompt,
            suffix=suffix,
            input_variables=input_variables,
        )


# In[ ]:




