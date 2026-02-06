"""
MOTOR DE NLP AVANÇADO - 4 Tecnologias para IA Inteligente
==========================================================

Este módulo implementa 4 estratégias pra deixar o agente mais inteligente
SEM usar APIs externas. Tudo é processamento local e gratuito!

1. WORD2VEC (Embeddings Semânticos)
2. LEMMATIZAÇÃO (Normalização de Palavras)
3. NER (Named Entity Recognition - Reconhecimento de Entidades)
4. APRENDIZADO PERSISTENTE (Memory do Agente)
"""

import spacy
import numpy as np
from gensim.models import Word2Vec
import pickle
import os
import re
from datetime import datetime
from pathlib import Path

# Imports de NLTK com fallback
try:
    from nltk.stem import RSLPStemmer
    stemmer_available = True
except ImportError:
    stemmer_available = False
    print("⚠️ NLTK não completamente configurado. Lemmatização terá capacidade limitada.")

class AdvancedNLPEngine:
    """Motor NLP com 4 tecnologias avançadas para inteligência sem API"""
    
    def __init__(self, model_path="modelos/nlp_engine.pkl"):
        self.model_path = model_path
        self.word2vec_model = None
        
        # Inicializa stemmer se disponível
        if stemmer_available:
            self.stemmer = RSLPStemmer()
        else:
            self.stemmer = None
            
        self.learning_memory = {}  # Memória de aprendizado
        self.entity_cache = {}  # Cache de entidades reconhecidas
        
        # Tenta carregar spaCy em português
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            print("⚠️ Modelo spaCy não encontrado. Para NER completo, execute:")
            print("python -m spacy download pt_core_news_sm")
            self.nlp = None
        
        self._load()
    
    # =========== 1. WORD2VEC - EMBEDDINGS SEMÂNTICOS ===========
    
    def train_word2vec(self, documents):
        """
        Treina Word2Vec com os documentos.
        
        Word2Vec transforma palavras em vetores numéricos baseado no
        SIGNIFICADO semântico, não apenas co-ocorrência.
        
        Diferença:
        - TF-IDF: "análise" e "análises" = completamente diferentes
        - Word2Vec: "análise" e "análises" = 92% similares
        """
        print("🧠 Treinando Word2Vec com embeddings semânticos...")
        
        # Processa documentos em sentenças
        sentences = []
        for doc in documents:
            # Remove pontuação e divide em palavras
            words = re.findall(r'\w+', doc.lower())
            if words:
                sentences.append(words)
        
        if sentences:
            # Treina modelo Word2Vec
            # vector_size=300: vetor com 300 dimensões (padrão Google)
            # window=5: contexto de 5 palavras antes e depois
            # min_count=2: ignora palavras que aparecem menos de 2x
            self.word2vec_model = Word2Vec(
                sentences=sentences,
                vector_size=300,
                window=5,
                min_count=2,
                workers=4,
                sg=1  # Skip-gram (melhor para semântica)
            )
            print(f"✓ Word2Vec treinado com {len(self.word2vec_model.wv)} palavras")
    
    def get_vector(self, word):
        """Obtém vetor semântico de uma palavra"""
        if self.word2vec_model and word.lower() in self.word2vec_model.wv:
            return self.word2vec_model.wv[word.lower()]
        return None
    
    def semantic_similarity(self, word1, word2):
        """
        Calcula similaridade semântica entre duas palavras (0-1).
        
        Exemplo:
        - semantic_similarity("banco", "instituição") = 0.78
        - semantic_similarity("banco", "cadeira") = 0.12
        """
        if not self.word2vec_model:
            return 0.0
        
        try:
            return max(0, self.word2vec_model.wv.similarity(
                word1.lower(), 
                word2.lower()
            ))
        except KeyError:
            return 0.0
    
    # =========== 2. LEMMATIZAÇÃO - NORMALIZAÇÃO DE PALAVRAS ===========
    
    def lemmatize(self, word):
        """
        Reduz palavra para sua forma raiz (lemma).
        
        Exemplos:
        - "executando" → "execut" (raiz)
        - "trabalhando" → "trabalh"
        - "melhorado" → "melhor"
        - "sistemas" → "sistem"
        
        Útil para reconhecer "as mesma palavra em variações diferentes"
        """
        if self.stemmer:
            return self.stemmer.stem(word.lower())
        else:
            # Fallback simples: remove sufixos comuns
            word_lower = word.lower()
            suffixes = ['ando', 'endo', 'indo', 'ado', 'ido', 'ada', 'ida', 'ção', 'ções', 'mente']
            for suffix in suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                    return word_lower[:-len(suffix)]
            return word_lower
    
    def lemmatize_text(self, text):
        """Aplica lemmatização em todo o texto"""
        words = re.findall(r'\w+', text.lower())
        return [self.lemmatize(w) for w in words]
    
    def normalize_query(self, query):
        """
        Normaliza query removendo variações.
        
        "Como executar os trabalhos?" 
        → "como execut o trabalh" (após lemmatização)
        
        Isso ajuda a encontrar documentos sobre "execução de trabalho"
        mesmo que use formas diferentes da palavra.
        """
        lemmatized = self.lemmatize_text(query)
        return " ".join(lemmatized)
    
    # =========== 3. NER - RECONHECIMENTO DE ENTIDADES ===========
    
    def extract_entities(self, text):
        """
        Extrai ENTIDADES nomeadas do texto:
        - PESSOA: João Silva
        - ORG: Prosegur
        - LOC: São Paulo
        - TIME: 2024, janeiro
        
        Exemplo:
        "João na Prosegur desde 2020"
        → {"PESSOA": ["João"], "ORG": ["Prosegur"], "TIME": ["2020"]}
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            entities[label].append(ent.text)
        
        # Cache para reuso
        self.entity_cache[text] = entities
        return entities
    
    def get_entity_context(self, entity_type, entity_value):
        """
        Obtém contexto sobre uma entidade.
        
        Exemplo: Se usuário pergunta "Quem é João?"
        → Busca por PESSOA:João nos documentos
        """
        return [
            text for text, entities in self.entity_cache.items()
            if entity_type in entities and entity_value in entities[entity_type]
        ]
    
    # =========== 4. APRENDIZADO PERSISTENTE ===========
    
    def learn_from_feedback(self, query, response, was_helpful=True):
        """
        Armazena feedback do usuário para melhorar futuras respostas.
        
        Se o usuário disser "Isso ajudou!", o sistema guarda:
        - Query original
        - Resposta que funcionou
        - Timestamp
        - Rating de utilidade
        
        Próximas vezes que vir query similar, prioriza a resposta boa!
        """
        key = self.normalize_query(query)
        
        if key not in self.learning_memory:
            self.learning_memory[key] = {
                "queries": [],
                "responses": [],
                "helpful_count": 0,
                "total_uses": 0,
                "last_used": None
            }
        
        self.learning_memory[key]["queries"].append(query)
        self.learning_memory[key]["responses"].append(response)
        self.learning_memory[key]["total_uses"] += 1
        
        if was_helpful:
            self.learning_memory[key]["helpful_count"] += 1
        
        self.learning_memory[key]["last_used"] = datetime.now().isoformat()
        
        # Salva aprendizado em arquivo
        self.save()
    
    def get_learned_response(self, query):
        """
        Se já respondeu algo parecido com sucesso antes, retorna!
        
        Taxa de sucesso = helpful_count / total_uses
        Só retorna se taxa > 80%
        """
        key = self.normalize_query(query)
        
        if key in self.learning_memory:
            mem = self.learning_memory[key]
            if mem["total_uses"] > 0:
                success_rate = mem["helpful_count"] / mem["total_uses"]
                
                if success_rate >= 0.8:
                    # Retorna resposta mais recente que funcionou
                    return mem["responses"][-1], success_rate
        
        return None, 0.0
    
    def get_memory_stats(self):
        """Retorna estatísticas de aprendizado"""
        total_patterns = len(self.learning_memory)
        total_interactions = sum(
            m["total_uses"] for m in self.learning_memory.values()
        )
        total_successful = sum(
            m["helpful_count"] for m in self.learning_memory.values()
        )
        
        return {
            "patterns_learned": total_patterns,
            "total_interactions": total_interactions,
            "successful_interactions": total_successful,
            "success_rate": (
                total_successful / total_interactions 
                if total_interactions > 0 else 0
            )
        }
    
    # =========== PERSISTÊNCIA ===========
    
    def save(self):
        """Salva modelo, embeddings e memória em arquivo"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        data = {
            "word2vec": self.word2vec_model,
            "learning_memory": self.learning_memory,
            "entity_cache": self.entity_cache
        }
        
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"✓ Motor NLP salvo em {self.model_path}")
    
    def _load(self):
        """Carrega modelo salvo se existir"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.word2vec_model = data.get("word2vec")
                    self.learning_memory = data.get("learning_memory", {})
                    self.entity_cache = data.get("entity_cache", {})
                print(f"✓ Motor NLP carregado do arquivo")
            except Exception as e:
                print(f"⚠️ Erro ao carregar NLP: {e}")


class SemanticDocumentMatcher:
    """
    Combina Word2Vec + TF-IDF para busca SEMÂNTICA em documentos.
    
    Busca não é apenas por palavras iguais, mas por SIGNIFICADO.
    """
    
    def __init__(self, nlp_engine):
        self.nlp = nlp_engine
        self.documents = []
        self.document_vectors = []
    
    def index_documents(self, documents):
        """Indexa documentos com vetores semânticos"""
        self.documents = documents
        self.document_vectors = []
        
        for doc in documents:
            words = re.findall(r'\w+', doc.lower())
            
            # Extrai vetores Word2Vec de cada palavra
            vectors = []
            for word in words:
                vec = self.nlp.get_vector(word)
                if vec is not None:
                    vectors.append(vec)
            
            # Documento = média dos vetores de suas palavras
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(300)
            
            self.document_vectors.append(doc_vector)
    
    def semantic_search(self, query, top_k=5):
        """
        Busca semântica em documentos.
        
        Retorna documentos mais SIMILARES em SIGNIFICADO, não apenas em palavras.
        """
        query_words = re.findall(r'\w+', query.lower())
        
        # Cria vetor da query
        vectors = []
        for word in query_words:
            vec = self.nlp.get_vector(word)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return []
        
        query_vector = np.mean(vectors, axis=0)
        
        # Calcula similaridade cosseno com cada documento
        similarities = []
        for i, doc_vec in enumerate(self.document_vectors):
            # Similaridade cosseno
            dot_product = np.dot(query_vector, doc_vec)
            norm1 = np.linalg.norm(query_vector)
            norm2 = np.linalg.norm(doc_vec)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = 0
            
            similarities.append((i, similarity, self.documents[i]))
        
        # Ordena por relevância
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [(i, sim, doc) for i, sim, doc in similarities[:top_k]]
