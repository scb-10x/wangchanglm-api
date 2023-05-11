from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow_hub as hub
import tensorflow_text

class Encoder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.array:
      """
        output dimension expected to be one dimension and normalized (unit vector)
      """
      ...


class MUSEEncoder(Encoder):
    def __init__(self, model_url: str = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
        self.embed = hub.load(model_url)

    def encode(self, texts: List[str]) -> np.array:
        embeds = self.embed(texts).numpy()
        embeds = embeds / np.linalg.norm(embeds, axis=1).reshape(embeds.shape[0], -1)
        return embeds


from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

@dataclass
class SensitiveTopic:
    name: str
    respond_message: str
    sensitivity: float = None # range from 0 to 1
    demonstrations: List[str] = None
    adhoc_embeded_demonstrations: np.array = None # dimension = [N_ADHOC, DIM]. Please kindly note that this suppose to 


DEFAULT_SENSITIVITY = 0.7


class SensitiveTopicProtector:
    def __init__(
        self,
        sensitive_topics: List[SensitiveTopic],
        encoder: Encoder = MUSEEncoder(),
        default_sensitivity: float = DEFAULT_SENSITIVITY
    ):
        self.sensitive_topics = sensitive_topics
        self.default_sensitivity = default_sensitivity
        self.encoder = encoder
        self.topic_embeddings = self._get_topic_embeddings()

    def _get_topic_embeddings(self) -> Dict[str, List[np.array]]:
        topic_embeddings = {}
        for topic in self.sensitive_topics:
            current_topic_embeddings = None
            if topic.demonstrations is not None:
                current_topic_embeddings = self.encoder.encode(texts=topic.demonstrations) if current_topic_embeddings is None \
                    else np.concatenate((current_topic_embeddings, self.encoder.encode(texts=topic.demonstrations)), axis=0)
            if topic.adhoc_embeded_demonstrations is not None:
                current_topic_embeddings = topic.adhoc_embeded_demonstrations if current_topic_embeddings is None \
                    else np.concatenate((current_topic_embeddings, topic.adhoc_embeded_demonstrations), axis=0)
            topic_embeddings[topic.name] = current_topic_embeddings
        return topic_embeddings

    def filter(self, text: str) -> Tuple[bool, str]:
        is_sensitive, respond_message = False, None
        text_embedding = self.encoder.encode([text,])
        for topic in self.sensitive_topics:
            risk_scores = np.einsum('ik,jk->j', text_embedding, self.topic_embeddings[topic.name])
            max_risk_score = np.max(risk_scores)
            if topic.sensitivity:
                if max_risk_score > (1.0 - topic.sensitivity):
                    return True, topic.respond_message
                continue
            if max_risk_score > (1.0 - self.default_sensitivity):
                return True, topic.respond_message
        return is_sensitive, respond_message
    
    @classmethod
    def fromRaw(cls, raw_sensitive_topics: List[Dict], encoder: Encoder = MUSEEncoder(), default_sensitivity: float = DEFAULT_SENSITIVITY):
        sensitive_topics = [SensitiveTopic(**topic) for topic in raw_sensitive_topics]
        return cls(sensitive_topics=sensitive_topics, encoder=encoder, default_sensitivity=default_sensitivity)


import pickle
import os

def loadGuardian():  
  # get current directory of this script
  current_directory = os.path.dirname(os.path.abspath(__file__))
  # load sensitive_topics.pkl in current_directory
  f = open(os.path.join(current_directory, "sensitive_topics.pkl"), "rb")
  sensitive_topics = pickle.load(f)
  f.close()

  guardian = SensitiveTopicProtector.fromRaw(sensitive_topics)
  return guardian

# ### Test Cases
# is_sensitive, respond_message = guardian.filter("หุ้่นตัวไหนมีแนวโน้มที่จะราคาขึ้นในเดือนถัดไป")
# assert is_sensitive == True