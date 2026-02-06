import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from advanced_nlp_engine import AdvancedNLPEngine, SemanticDocumentMatcher

PORTUGUESE_STOPWORDS = {
    "o","a","os","as","de","do","da","dos","das","em","para","por","com",
    "um","uma","uns","umas","e","ou","não","que","se","na","no","nas","nos",
    "ao","aos","à","às","como","mais","menos","já","também","ser","são"
}

class KnowledgeModel:
    def __init__(self, model_path="modelos/model.pkl"):
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            min_df=1,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words=list(PORTUGUESE_STOPWORDS)
        )
        self.tfidf_matrix = None
        self.documents = []
        
        # Inicializa motor NLP avançado
        self.nlp_engine = AdvancedNLPEngine()
        self.semantic_matcher = SemanticDocumentMatcher(self.nlp_engine)

    def train(self, documents):
        self.documents = documents
        if documents:
            # Treina TF-IDF (busca por palavras-chave)
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Treina Word2Vec (busca semântica)
            self.nlp_engine.train_word2vec(documents)
            
            # Indexa documentos para busca semântica
            self.semantic_matcher.index_documents(documents)
            
            print(f"✓ Modelo treinado com:")
            print(f"  - TF-IDF: {len(documents)} documentos")
            print(f"  - Word2Vec: Embeddings semânticos")
            print(f"  - NER: Reconhecimento de entidades")
            print(f"  - Memory: Sistema de aprendizado")

    def query(self, text, top_k=5, use_semantic=True):
        """
        Busca em documentos com 2 métodos combinados:
        1. TF-IDF (busca por palavras-chave)
        2. Word2Vec (busca por significado semântico)
        """
        if self.tfidf_matrix is None or not self.documents:
            return []
        
        # Primeiro tenta retornar resposta do aprendizado anterior
        learned_response, success_rate = self.nlp_engine.get_learned_response(text)
        if learned_response and success_rate >= 0.9:
            print(f"💡 Usando resposta aprendida (confiança: {success_rate:.0%})")
            # Encontra índice do documento aprendido
            for i, doc in enumerate(self.documents):
                if doc == learned_response:
                    return [(i, 1.0, doc)]
        
        # Busca por TF-IDF (método original)
        q_vec = self.vectorizer.transform([text])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        tfidf_top_idx = tfidf_scores.argsort()[::-1][:top_k]
        
        # Busca semântica (Word2Vec) se habilitado
        results = []
        if use_semantic and self.nlp_engine.word2vec_model:
            semantic_results = self.semantic_matcher.semantic_search(text, top_k)
            
            # Combina resultados TF-IDF + Word2Vec
            combined_scores = {}
            
            for idx in tfidf_top_idx:
                combined_scores[idx] = tfidf_scores[idx] * 0.6  # TF-IDF tem 60% peso
            
            for idx, sem_score, _ in semantic_results:
                if idx in combined_scores:
                    combined_scores[idx] += sem_score * 0.4  # Word2Vec tem 40% peso
                else:
                    combined_scores[idx] = sem_score * 0.4
            
            # Ordena por score combinado
            sorted_results = sorted(
                combined_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            results = [
                (i, float(score), self.documents[i]) 
                for i, score in sorted_results[:top_k]
            ]
        else:
            # Retorna apenas resultados TF-IDF
            results = [
                (i, float(tfidf_scores[i]), self.documents[i]) 
                for i in tfidf_top_idx
            ]
        
        # Extrai entidades da query para contexto
        entities = self.nlp_engine.extract_entities(text)
        if entities:
            print(f"🏷️ Entidades detectadas: {entities}")
        
        return results
    
    def record_feedback(self, query, response, was_helpful=True):
        """Registra feedback do usuário para aprendizado contínuo"""
        self.nlp_engine.learn_from_feedback(query, response, was_helpful)

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self, f)
        
        # Salva o motor NLP separadamente
        if hasattr(self, 'nlp_engine'):
            self.nlp_engine.save()

    @staticmethod
    def load(model_path="modelos/model.pkl"):
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                
                # Garante compatibilidade com modelos antigos
                # Inicializa novos atributos se não existirem
                if not hasattr(model, 'nlp_engine'):
                    model.nlp_engine = AdvancedNLPEngine()
                    print("⚠️ Modelo antigo detectado - Inicializando motor NLP")
                
                if not hasattr(model, 'semantic_matcher'):
                    model.semantic_matcher = SemanticDocumentMatcher(model.nlp_engine)
                    print("⚠️ Inicializando busca semântica")
                
                return model
        return None