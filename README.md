# Proyecto RAG - TFG

## Descripcion
Proyecto de exploracion y comparacion entre los frameworks **LangChain** y **LlamaIndex** para la implementacion de un sistema RAG (Retrieval Augmented Generation), como base para el Trabajo de Fin de Grado.

## Stack tecnologico

- **Python 3.12.7**  gestionado con `pyenv`, aislado de la version 3.14 del sistema por compatibilidad con librerias de ML
- **Ollama**  motor para ejecutar modelos de lenguaje localmente, sin dependencia de APIs de pago
  - `qwen2.5:1.5b`  modelo de generacion de texto (LLM)
  - `nomic-embed-text`  modelo de embeddings, especializado en tareas de retrieval
- **LangChain** + **langchain-community** + **langchain-ollama**
- **LlamaIndex** + **llama-index-llms-ollama** + **llama-index-embeddings-ollama**

## Por que estas decisiones tecnicas

- **Modelos locales via Ollama** en vez de APIs de pago: sin coste, reproducible, sin enviar datos a terceros apropiado para un entorno academico.
- **nomic-embed-text**: modelo ligero (274 MB) disenado especificamente para retrieval, con ventana de contexto de 8192 tokens, disponible directamente en Ollama.
- **qwen2.5:1.5b**: modelo pequeno optimizado para CPU (sin GPU dedicada disponible), buen seguimiento de instrucciones para tareas de RAG.

## Instalacion

### Requisitos
- WSL2 con Ubuntu
- pyenv
- Ollama

### Pasos
\`\`\`bash
pyenv local 3.12.7
pip install langchain langchain-community llama-index
pip install langchain-ollama llama-index-llms-ollama llama-index-embeddings-ollama
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
\`\`\`

## Estado actual
- [x] Entorno configurado (Python, Git, GitHub)
- [x] Ollama instalado con modelos de LLM y embeddings
- [ ] Implementacion RAG con LangChain
- [ ] Implementacion RAG con LlamaIndex
- [ ] Comparacion de resultados