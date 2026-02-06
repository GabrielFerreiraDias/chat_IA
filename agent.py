from knowledge_model import KnowledgeModel
import re
from collections import Counter
from llm_engine import LocalLLMEngine, HybridResponseGenerator

class AIAgent:
    def __init__(self, document_processor, model_path="modelos/model.pkl", use_llm=True):
        self.document_processor = document_processor
        self.knowledge_model = KnowledgeModel.load(model_path) or KnowledgeModel(model_path)
        
        # 🧠 Inicializa LLM local para respostas mais naturais
        self.llm_engine = LocalLLMEngine(enabled=use_llm)
        self.hybrid_generator = HybridResponseGenerator(self.llm_engine)
        self.use_llm = use_llm and self.llm_engine.is_available()
        
        if self.use_llm:
            print("✅ Chat com LLM local ativado - respostas naturais!")
        else:
            print("📝 Chat em modo tradicional - sem LLM")

        self.greetings = {
            "oi": "Olá! Sou o agente de IA da melhoria contínua. Como posso ajudá-lo?",
            "ola": "Olá! Sou o agente de IA da melhoria contínua. Como posso ajudá-lo?",
            "olá": "Olá! Sou o agente de IA da melhoria contínua. Como posso ajudá-lo?",
            "opa": "Opa! Sou o agente de IA da melhoria contínua. Como posso ajudá-lo?",
            "hey": "Hey! Sou o agente de IA da melhoria contínua. Como posso ajudá-lo?",
            "e ai": "E aí! Sou o agente de IA da melhoria contínua. Como posso ajudá-lo?",
            "tudo bem": "Tudo bem sim! Sou o agente de IA da melhoria contínua. Em que posso ajudá-lo?",
            "obrigado": "De nada! Fico feliz em ajudar.",
            "valeu": "De nada! Fico feliz em ajudar.",
            "tchau": "Até logo!"
        }
        
        # Respostas para perguntas sobre identidade/pessoais
        self.identity_patterns = {
            r"(quem|qual) (?:é|sou) (?:você|vc|seu nome)": "Sou um assistente de IA criado para ajudar com informações. Estou aqui para responder suas dúvidas sobre os tópicos da minha base de conhecimento!",
            r"(qual|quem) (?:são|sao|somos) (?:os )?chefes": "Essa informação específica não está documentada para mim, mas você pode verificar o organograma da empresa ou conversar com seu gestor direto!",
            r"(quem|qual) (?:é|são) (?:você|vc|seu criador|seu desenvolvedor)": "Fui desenvolvido como um assistente inteligente para esta empresa. Minha função é tornar o acesso à informação mais rápido e fácil!",
            r"(qual|quem) é (?:meu|seu) (?:chefe|supervisor|gerente)": "Essa informação é mais adequada para ser obtida no sistema de RH ou com seu gestor direto. Posso ajudar com outras informações?",
            r"(como|aonde) (?:posso )?(?:te )?encontrar|(?:qual|onde) (?:é|fica) (?:você|vc)": "Estou aqui neste chat, disponível 24/7 para ajudar! Você também pode consultar a documentação armazenada em minha base de conhecimento.",
            r"(qual|quem) (?:trabalha|trabalham) (?:com|aqui)": "Ótima pergunta! Mas para informações sobre membros da equipe, recomendo consultar o sistema interno ou checar o diretório da empresa.",
        }

    def _is_greeting(self, message):
        msg = re.sub(r'[^\w\s]', '', message.lower().strip())
        return any(g in msg for g in self.greetings.keys())

    def _respond_greeting(self, message):
        msg = re.sub(r'[^\w\s]', '', message.lower().strip())
        for g, r in self.greetings.items():
            if g in msg:
                return r
        return "Olá! Como posso ajudá-lo?"
    
    def _is_identity_question(self, message):
        """Detecta perguntas sobre identidade, pessoais ou fora do escopo"""
        msg = message.lower()
        return any(re.search(pattern, msg) for pattern in self.identity_patterns.keys())
    
    def _respond_identity_question(self, message):
        """Responde perguntas sobre identidade/pessoais de forma humanizada"""
        msg = message.lower()
        for pattern, response in self.identity_patterns.items():
            if re.search(pattern, msg):
                return response
        # Resposta padrão se não encontrar padrão específico
        return "Essa é uma ótima pergunta! Infelizmente, essa informação não está na minha base de conhecimento, mas posso ajudar com outras dúvidas!"

    def _keywords(self, text):
        words = re.findall(r'\w+', text.lower())
        stop = {"o","a","os","as","de","do","da","dos","das","em","para","por","com","um","uma","e","ou","não","que","se","na","no","nas","nos"}
        return [w for w in words if w not in stop and len(w) > 3]

    def _best_sentences(self, docs, query):
        keywords = self._keywords(query)
        sentences = []
        for _, _, doc in docs:
            parts = re.split(r'[.!?]+', doc)
            for s in parts:
                s = s.strip()
                if len(s) < 40:
                    continue
                score = sum(1 for k in keywords if k in s.lower())
                if score > 0:
                    sentences.append((score, s))
        sentences.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in sentences[:8]]

    def train_and_save_model(self):
        docs = [c["text"] for c in self.document_processor.get_chunks()]
        self.knowledge_model.train(docs)
        self.knowledge_model.save()
        print(f"✓ Modelo treinado com {len(docs)} chunks")

    def _reformat_response(self, text):
        """Reformula a resposta para ser mais natural e fluida"""
        # 1. Quebra em sentenças
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        
        if not sentences:
            return text
        
        # 2. Remove duplicatas mantendo ordem
        seen = set()
        unique_sents = []
        for s in sentences:
            s_normalized = s.lower().strip()
            if s_normalized not in seen:
                seen.add(s_normalized)
                unique_sents.append(s)
        
        if not unique_sents:
            return text
        
        # 3. Ordena por tamanho (prioriza sentenças mais informativas)
        sorted_sents = sorted(unique_sents, key=lambda x: len(x), reverse=True)[:5]
        
        # 4. Conectivos naturais para fluir melhor entre sentenças
        connectors = [
            "Além disso,",
            "Vale mencionar que",
            "É importante destacar que", 
            "Somando a isso,",
            "Também podemos notar que",
            "Com relação a isso,",
            "Nesse contexto,"
        ]
        
        # 5. Constrói resposta mais natural
        response = sorted_sents[0] + "."
        
        for i, sent in enumerate(sorted_sents[1:], 1):
            connector = connectors[i % len(connectors)]
            response += f" {connector} {sent.lower() if sent[0].isupper() else sent}."
        
        return response

    def chat(self, message):
        # 1. Verifica se é saudação - SEMPRE usa resposta pré-definida
        if self._is_greeting(message):
            return self._respond_greeting(message)
        
        # 2. Verifica se é pergunta sobre identidade/pessoal
        if self._is_identity_question(message):
            return self._respond_identity_question(message)

        # 3. Busca nos documentos com Word2Vec + TF-IDF
        results = self.knowledge_model.query(message, top_k=5)
        if not results:
            return "Desculpa, não encontrei informações suficientes nos documentos."

        # 4. Extrai documentos relevantes
        relevant_docs = []
        for idx, score, doc in results:
            if score > 0.1:  # Apenas documentos com relevância mínima
                relevant_docs.append(doc.strip())
        
        if not relevant_docs:
            return "Não encontrei um trecho claro nos documentos. Tente ser mais específico."
        
        # 5. 🧠 MODO HÍBRIDO: Tenta LLM primeiro, depois fallback tradicional
        if self.use_llm:
            # LLM reformula resposta baseado nos documentos encontrados
            response, used_llm = self.hybrid_generator.generate_response(
                message, 
                relevant_docs, 
                use_llm=True
            )
            if used_llm:
                return response
        
        # Fallback: Modo tradicional (extração de sentenças + reformulação)
        best_sents = self._best_sentences(results, message)
        if best_sents:
            raw_response = " ".join(best_sents)
            return self._reformat_response(raw_response)
        
        # Último recurso: retorna docs concatenados e reformatados
        raw_response = "\n\n".join(relevant_docs[:2])
        return self._reformat_response(raw_response)