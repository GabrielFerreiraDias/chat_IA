"""
TESTE RÁPIDO - Sistema Completo com LLM Local
==============================================

Este script testa:
1. Carregamento do modelo treinado
2. Busca com Word2Vec + TF-IDF
3. LLM local (se disponível)
4. Sistema híbrido de respostas
"""

import os
import sys

def test_basic_imports():
    """Testa imports básicos"""
    print("\n" + "="*60)
    print("1️⃣ TESTANDO IMPORTS BÁSICOS")
    print("="*60)
    
    try:
        from knowledge_model import KnowledgeModel
        print("✅ KnowledgeModel importado")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False
    
    try:
        from agent import AIAgent
        print("✅ AIAgent importado")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False
    
    try:
        from llm_engine import LocalLLMEngine, HybridResponseGenerator
        print("✅ LLM Engine importado")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False
    
    return True

def test_model_loading():
    """Testa carregamento do modelo treinado"""
    print("\n" + "="*60)
    print("2️⃣ TESTANDO CARREGAMENTO DO MODELO")
    print("="*60)
    
    try:
        from knowledge_model import KnowledgeModel
        
        if not os.path.exists("modelos/model.pkl"):
            print("⚠️ Modelo não encontrado!")
            print("💡 Execute: python train_model.py")
            return False
        
        model = KnowledgeModel.load("modelos/model.pkl")
        print("✅ Modelo carregado com sucesso")
        
        # Verifica componentes
        if hasattr(model, 'nlp_engine'):
            print("✅ NLP Engine presente")
        else:
            print("⚠️ NLP Engine não encontrado (treinar novamente?)")
        
        if hasattr(model, 'vectorizer'):
            print("✅ TF-IDF Vectorizer presente")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

def test_llm_availability():
    """Testa disponibilidade do LLM local"""
    print("\n" + "="*60)
    print("3️⃣ TESTANDO LLM LOCAL")
    print("="*60)
    
    try:
        from llm_engine import LocalLLMEngine
        
        # Verifica bibliotecas LLM disponíveis
        has_llama_cpp = False
        has_ctransformers = False
        
        try:
            import llama_cpp
            print("✅ llama-cpp-python instalado")
            has_llama_cpp = True
        except ImportError:
            print("⚠️ llama-cpp-python NÃO instalado")
        
        try:
            import ctransformers
            print("✅ ctransformers instalado")
            has_ctransformers = True
        except ImportError:
            print("⚠️ ctransformers NÃO instalado")
        
        if not has_llama_cpp and not has_ctransformers:
            print("\n❌ Nenhuma biblioteca LLM instalada!")
            print("💡 Instale uma das opções:")
            print("   - pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
            print("   - pip install ctransformers")
            return False
        
        # Verifica se há modelo .gguf
        model_dir = "modelos"
        gguf_files = []
        if os.path.exists(model_dir):
            gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
        
        if gguf_files:
            print(f"✅ Modelo(s) .gguf encontrado(s): {', '.join(gguf_files)}")
        else:
            print("⚠️ Nenhum modelo .gguf encontrado em modelos/")
            print("💡 Baixe Phi-3-mini: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
            return False
        
        # Tenta inicializar LLM
        print("\n⏳ Tentando carregar LLM (pode levar alguns segundos)...")
        llm = LocalLLMEngine(enabled=True)
        
        if llm.is_available():
            print("✅ LLM local DISPONÍVEL e FUNCIONANDO!")
            return True
        else:
            print("⚠️ LLM local não disponível")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar LLM: {e}")
        return False

def test_query_without_llm():
    """Testa query sem LLM (modo tradicional)"""
    print("\n" + "="*60)
    print("4️⃣ TESTANDO QUERY SEM LLM (Tradicional)")
    print("="*60)
    
    try:
        from document_processador import DocumentProcessor
        from agent import AIAgent
        
        dp = DocumentProcessor("aprendizado")
        dp.process_all_documents()
        
        # Cria agente SEM LLM
        agent = AIAgent(dp, use_llm=False)
        
        # Testa query
        query = "Como funciona o controle de documentos?"
        print(f"\n📝 Query: {query}")
        
        response = agent.chat(query)
        print(f"\n🤖 Resposta (sem LLM):\n{response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_with_llm():
    """Testa query COM LLM (modo híbrido)"""
    print("\n" + "="*60)
    print("5️⃣ TESTANDO QUERY COM LLM (Híbrido)")
    print("="*60)
    
    try:
        from document_processador import DocumentProcessor
        from agent import AIAgent
        
        dp = DocumentProcessor("aprendizado")
        dp.process_all_documents()
        
        # Cria agente COM LLM
        agent = AIAgent(dp, use_llm=True)
        
        if not agent.use_llm:
            print("⚠️ LLM não está disponível")
            print("💡 Verifique etapa anterior (3️⃣)")
            return False
        
        # Testa query
        query = "Como funciona o controle de documentos?"
        print(f"\n📝 Query: {query}")
        
        print("\n⏳ Gerando resposta com LLM (pode levar 1-3 segundos)...")
        response = agent.chat(query)
        print(f"\n🧠 Resposta (com LLM):\n{response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes"""
    print("\n" + "="*60)
    print("🧪 TESTE COMPLETO DO SISTEMA COM LLM LOCAL")
    print("="*60)
    
    results = {}
    
    # Teste 1: Imports
    results['imports'] = test_basic_imports()
    
    # Teste 2: Modelo
    results['model'] = test_model_loading()
    
    # Teste 3: LLM
    results['llm'] = test_llm_availability()
    
    # Teste 4: Query tradicional
    if results['model']:
        results['query_traditional'] = test_query_without_llm()
    else:
        results['query_traditional'] = False
        print("\n⚠️ Pulando teste de query tradicional (modelo não carregado)")
    
    # Teste 5: Query com LLM
    if results['model'] and results['llm']:
        results['query_llm'] = test_query_with_llm()
    else:
        results['query_llm'] = False
        print("\n⚠️ Pulando teste de query com LLM (pré-requisitos não atendidos)")
    
    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)
    
    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name.upper()}: {'PASSOU' if passed else 'FALHOU'}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*60)
    print(f"📈 RESULTADO FINAL: {passed}/{total} testes passaram")
    print("="*60)
    
    if passed == total:
        print("\n🎉 TUDO FUNCIONANDO PERFEITAMENTE!")
        print("✅ Sistema pronto para produção com LLM local")
        print("\n💡 Próximo passo: python app.py")
    elif results.get('query_traditional'):
        print("\n✅ Sistema FUNCIONAL (modo tradicional)")
        print("⚠️ LLM local não disponível, mas tudo funciona normalmente")
        print("\n💡 Para ativar LLM:")
        print("   1. Certifique-se que tem uma biblioteca LLM:")
        print("      - ctransformers (JÁ instalado? ✅)")
        print("      - OU llama-cpp-python")
        print("   2. Baixe um modelo .gguf:")
        print("      - TinyLlama (800MB): https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        print("      - Phi-3-mini (2GB): https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
        print("   3. Coloque o arquivo .gguf em modelos/")
        print("   4. Execute este script novamente")
        print("\n📚 Guia completo: BAIXAR_MODELO.md")
    else:
        print("\n❌ Sistema com problemas")
        print("💡 Verifique os testes falhados acima")
        print("📚 Consulte GUIA_LLM_LOCAL.md para ajuda")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
