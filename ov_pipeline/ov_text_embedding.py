# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import Any, List, Optional

from numpy import array, ndarray

from typing import Optional, Union, Any, List
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer
from pathlib import Path
import openvino as ov
import numpy as np
from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.ai_service_client_base import AIServiceClientBase
from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)

logger: logging.Logger = logging.getLogger(__name__)


class OpenVINOTextEmbedding(EmbeddingGeneratorBase, AIServiceClientBase):
    model: Any
    tokenizer: Any
    do_norm: bool
    num_stream: int

    def __init__(
        self,
        ai_model_id: str,
        do_norm: bool = True,
        ov_config: Optional[dict] = {
            "device_name": "CPU",
            "config": {"PERFORMANCE_HINT": "THROUGHPUT"},
        },
        model_kwargs: Optional[dict] = {
            "model_max_length": 512,
        },
    ) -> None:
        """
        Initializes a new instance of the HuggingFaceTextEmbedding class.

        Arguments:
            ai_model_id {str} -- Hugging Face model card string
            do_norm {bool} -- whether do normalization
            ov_config {Optional[dict]} -- openvino configuation
            model_kwargs {Optional[dict]} -- Additional dictionary of keyword arguments
                passed along to the tokenizer's `from_pretrained(..., **model_kwargs)` function.

        Note that this model will be downloaded from the Hugging Face model hub.
        """
        _model_kwargs = model_kwargs or {}
        _ov_config = ov_config or {}
        tokenizer = AutoTokenizer.from_pretrained(ai_model_id, **_model_kwargs)
        core = ov.Core()
        model_path = Path(ai_model_id) / "openvino_model.xml"
        model = core.compile_model(model_path, **_ov_config)
        num_stream = model.get_property('NUM_STREAMS')
        super().__init__(
            ai_model_id=ai_model_id,
            model=model,
            tokenizer=tokenizer,
            do_norm=do_norm,
            num_stream=num_stream
        )

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        # Empty string or list of ints
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            # Sum of length of individual strings
            return sum([len(t) for t in text])

    def encode(self, sentences: Union[str, List[str]]):
        """
        Computes sentence embeddings

        Args: 
            sentences: the sentences to embed

        Returns:
           By default, a list of tensors is returned.
        """
        all_embeddings = []
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        nireq = self.num_stream + 1
        infer_queue = ov.AsyncInferQueue(self.model, nireq)

        def postprocess(request, userdata):
            embeddings = request.get_output_tensor(0).data
            embeddings = np.mean(embeddings, axis=1)
            if self.do_norm:
                embeddings = normalize(embeddings, 'l2')
            all_embeddings.extend(embeddings)

        infer_queue.set_callback(postprocess)

        for i, sentence in enumerate(sentences_sorted):
            inputs = {}
            features = self.tokenizer(
                sentence, padding=True, truncation=True, return_tensors='np')
            for key in features:
                inputs[key] = features[key]
            infer_queue.start_async(inputs, i)
        infer_queue.wait_all()
        all_embeddings = np.asarray(all_embeddings)
        return all_embeddings

    async def generate_embeddings(self, texts: List[str]) -> ndarray:
        """
        Generates embeddings for a list of texts.

        Arguments:
            texts {List[str]} -- Texts to generate embeddings for.

        Returns:
            ndarray -- Embeddings for the texts.
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.encode(texts)
            return array(embeddings)
        except Exception as e:
            raise AIException("OpenVINO embeddings failed", e)
