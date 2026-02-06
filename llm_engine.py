"""
LLM LOCAL ENGINE - Reformulação Natural de Respostas
====================================================

Este módulo integra um LLM local (via llama-cpp-python OU ctransformers) para deixar
as respostas do chat mais naturais e humanas.

Sistema Híbrido:
1. Busca documentos relevantes (Word2Vec + TF-IDF) - RÁPIDO
2. LLM reformula resposta baseado nos docs - NATURAL
3. Fallback automático se LLM falhar - CONFIÁVEL

Bibliotecas suportadas:
- llama-cpp-python (preferida, mais features)
- ctransformers (alternativa, mais fácil de instalar)
"""

import os
from pathlib import Path

class LocalLLMEngine:
    """Motor de LLM local para reformulação natural de respostas"""
    
    def __init__(self, model_path=None, enabled=True):
        """
        Inicializa o motor de LLM local
        
        Args:
            model_path: Caminho para o modelo .gguf (opcional)
            enabled: Se True, tenta carregar o LLM. Se False, usa apenas fallback
        """
        self.llm = None
        self.enabled = enabled
        self.model_loaded = False
        self.backend = None  # 'llama-cpp' ou 'ctransformers'
        
        if not enabled:
            print("🔧 LLM local desabilitado - usando modo tradicional")
            return
        
        # Tenta carregar com llama-cpp-python primeiro
        if self._try_llama_cpp(model_path):
            return
        
        # Se falhar, tenta ctransformers
        if self._try_ctransformers(model_path):
            return
        
        # Nenhuma biblioteca disponível
        print("⚠️ Nenhuma biblioteca LLM disponível")
        print("💡 Instale uma das opções:")
        print("   - pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
        print("   - pip install ctransformers")
    
    def _try_llama_cpp(self, model_path):
        """Tenta carregar com llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            # Se não especificar modelo, tenta encontrar na pasta modelos/
            if model_path is None:
                model_dir = Path("modelos")
                if model_dir.exists():
                    # Procura por arquivos .gguf
                    gguf_files = list(model_dir.glob("*.gguf"))
                    if gguf_files:
                        model_path = str(gguf_files[0])
                        print(f"📦 Modelo encontrado: {model_path}")
                    else:
                        return False
            
            if model_path and os.path.exists(model_path):
                print(f"🚀 Carregando LLM local (llama-cpp-python): {model_path}")
                print("   ⏳ Isso pode levar alguns segundos...")
                
                # Carrega modelo com configurações otimizadas
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,        # Contexto de 2048 tokens
                    n_threads=4,       # Usa 4 threads
                    n_gpu_layers=0,    # 0 = apenas CPU (mude se tiver GPU)
                    verbose=False
                )
                
                self.backend = 'llama-cpp'
                self.model_loaded = True
                print("✅ LLM local carregado com sucesso! (backend: llama-cpp-python)")
                print("🧠 Respostas agora serão mais naturais e humanas")
                return True
                
        except ImportError:
            return False
        except Exception as e:
            print(f"⚠️ Erro ao carregar com llama-cpp-python: {e}")
            return False
        
        return False
    
    def _try_ctransformers(self, model_path):
        """Tenta carregar com ctransformers"""
        try:
            from ctransformers import AutoModelForCausalLM
            
            # Se não especificar modelo, usa modelo padrão pequeno
            if model_path is None:
                model_dir = Path("modelos")
                if model_dir.exists():
                    gguf_files = list(model_dir.glob("*.gguf"))
                    if gguf_files:
                        model_path = str(gguf_files[0])
                        print(f"📦 Modelo encontrado: {model_path}")
                    else:
                        print("⚠️ Nenhum modelo .gguf em modelos/")
                        print("💡 Baixe um modelo:")
                        print("   Phi-3-mini: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
                        print("   TinyLlama: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
                        return False
            
            if model_path and os.path.exists(model_path):
                print(f"🚀 Carregando LLM local (ctransformers): {model_path}")
                print("   ⏳ Isso pode levar alguns segundos...")
                
                self.llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    model_type='llama',  # Funciona para LLaMA, Phi, Mistral
                    context_length=2048,
                    threads=4
                )
                
                self.backend = 'ctransformers'
                self.model_loaded = True
                print("✅ LLM local carregado com sucesso! (backend: ctransformers)")
                print("🧠 Respostas agora serão mais naturais e humanas")
                return True
                
        except ImportError:
            return False
        except Exception as e:
            print(f"⚠️ Erro ao carregar com ctransformers: {e}")
            return False
        
        return False
    def is_available(self):
        """Verifica se o LLM está disponível"""
        return self.enabled and self.model_loaded and self.llm is not None
    
    def reformulate_response(self, query, context_docs, max_tokens=300):
        """
        Reformula a resposta usando o LLM local
        
        Args:
            query: Pergunta do usuário
            context_docs: Lista de documentos relevantes encontrados
            max_tokens: Máximo de tokens na resposta
            
        Returns:
            str: Resposta reformulada naturalmente
        """
        if not self.is_available():
            # Fallback: retorna documentos concatenados
            return " ".join(context_docs[:3])
        
        try:
            # Cria prompt para o LLM
            context = "\n\n".join([f"Documento {i+1}: {doc[:500]}" 
                                   for i, doc in enumerate(context_docs[:3])])
            
            prompt = f"""Você é um assistente útil e prestativo. Com base nos documentos fornecidos, responda a pergunta do usuário de forma clara, natural e amigável.

Documentos de Referência:
{context}

Pergunta do Usuário: {query}

Instruções:
- Responda em português brasileiro
- Seja claro e objetivo
- Use um tom amigável e profissional
- Base sua resposta APENAS nos documentos fornecidos
- Se não souber, diga que não tem informação suficiente
- Não invente informações

Resposta:"""
            
            # Gera resposta de acordo com o backend
            if self.backend == 'llama-cpp':
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stop=["Pergunta:", "Usuário:", "\n\n\n"],
                    echo=False
                )
                answer = response['choices'][0]['text'].strip()
                
            elif self.backend == 'ctransformers':
                answer = self.llm(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    stop=["Pergunta:", "Usuário:", "\n\n\n"]
                )
            else:
                return " ".join(context_docs[:3])
            
            # Remove possíveis prefixos indesejados
            prefixes = ["Resposta:", "Assistente:", "IA:"]
            for prefix in prefixes:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
            
            return answer
            
        except Exception as e:
            print(f"⚠️ Erro ao gerar resposta com LLM: {e}")
            # Fallback: retorna documentos concatenados
            return " ".join(context_docs[:3])
    
    def quick_answer(self, query, max_tokens=150):
        """
        Gera uma resposta rápida sem documentos de contexto
        (útil para saudações e perguntas gerais)
        """
        if not self.is_available():
            return None
        
        try:
            prompt = f"""Você é um assistente prestativo. Responda brevemente em português.

Usuário: {query}
Assistente:"""
            
            # Gera resposta de acordo com o backend
            if self.backend == 'llama-cpp':
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    stop=["Usuário:", "\n\n"],
                    echo=False
                )
                return response['choices'][0]['text'].strip()
                
            elif self.backend == 'ctransformers':
                answer = self.llm(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    stop=["Usuário:", "\n\n"]
                )
                return answer.strip()
            
        except Exception as e:
            print(f"⚠️ Erro na resposta rápida: {e}")
            return None


class HybridResponseGenerator:
    """
    Gerador híbrido que combina:
    - Busca tradicional (rápida e precisa)
    - LLM local (natural e humano)
    """
    
    def __init__(self, llm_engine=None):
        """
        Inicializa o gerador híbrido
        
        Args:
            llm_engine: Instância de LocalLLMEngine (opcional)
        """
        self.llm = llm_engine or LocalLLMEngine()
    
    def generate_response(self, query, relevant_docs, use_llm=True):
        """
        Gera resposta usando abordagem híbrida
        
        Args:
            query: Pergunta do usuário
            relevant_docs: Documentos relevantes encontrados
            use_llm: Se True, tenta usar LLM. Se False, usa método tradicional
            
        Returns:
            tuple: (resposta, usado_llm)
        """
        # Se LLM disponível e habilitado, usa ele
        if use_llm and self.llm.is_available():
            print("🧠 Gerando resposta com LLM local...")
            response = self.llm.reformulate_response(query, relevant_docs)
            return response, True
        
        # Fallback: método tradicional (concatena documentos)
        print("📝 Usando método tradicional...")
        response = " ".join(relevant_docs[:3])
        return response, False
    
    def get_stats(self):
        """Retorna estatísticas sobre o uso do LLM"""
        return {
            "llm_available": self.llm.is_available(),
            "llm_loaded": self.llm.model_loaded,
            "llm_enabled": self.llm.enabled
        }


# Exemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO LLM LOCAL ENGINE")
    print("=" * 60)
    
    # Inicializa LLM
    llm = LocalLLMEngine()
    
    if llm.is_available():
        print("\n✅ LLM disponível! Testando...")
        
        # Teste com documentos
        query = "Como funciona o controle de documentos?"
        docs = [
            "O controle de documentos segue o procedimento PROMC_PR_1.2.1. "
            "Primeiro, o documento é criado no sistema. Depois, passa por revisão "
            "e aprovação antes de ser publicado.",
            "Documentos devem ser versionados corretamente e manter histórico "
            "de alterações para rastreabilidade."
        ]
        
        response = llm.reformulate_response(query, docs)
        print(f"\n📝 Pergunta: {query}")
        print(f"\n🤖 Resposta LLM:\n{response}")
        
    else:
        print("\n⚠️ LLM não disponível")
        print("💡 Baixe um modelo .gguf e coloque em modelos/")
        print("   Recomendado: Phi-3-mini (2GB, rápido)")
        print("   Link: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
