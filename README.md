# Proyecto RAG - TFG

## Descripcion
Proyecto de exploracion y comparacion entre los frameworks **LangChain** y **LlamaIndex** para la implementacion de un sistema RAG (Retrieval Augmented Generation), como base para el Trabajo de Fin de Grado.

## Stack tecnologico

- **Python 3.12.7**  gestionado con `pyenv`, aislado de la version 3.14 del sistema por compatibilidad con librerias de ML
- **Ollama**  motor para ejecutar modelos de lenguaje localmente, sin dependencia de APIs de pago
  - `qwen2.5:1.5b`  modelo de generacion de texto (LLM)
  - `nomic-embed-text`  modelo de embeddings, especializado en tareas de retrieval
- **LangChain** + **langchain-ollama**
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
python -m pip install -r requirements.txt
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
\`\`\`

## Primer RAG con LangChain

La implementacion inicial esta en `src/rag_langchain.py` y usa Ollama para
ejecutar los modelos localmente:

\`\`\`bash
cp .env.example .env
python src/rag_langchain.py "Que significa RAG?"
\`\`\`

El flujo sigue cinco pasos:

1. **Carga**: el programa lee los ficheros Markdown de `data/` como documentos
  LangChain.
2. **Fragmentacion**: `RecursiveCharacterTextSplitter` divide los documentos
  en fragmentos pequenos con solapamiento para conservar contexto.
3. **Embeddings**: `OllamaEmbeddings` convierte cada fragmento en un vector
  numerico usando `nomic-embed-text`.
4. **Retrieval**: `InMemoryVectorStore` busca los fragmentos mas parecidos a la
  pregunta y el retriever devuelve los `k` mejores resultados.
5. **Generation**: `ChatOllama` recibe un prompt con el contexto recuperado y
  genera una respuesta con `qwen2.5:1.5b`.

LangChain aporta las piezas y sus interfaces composables; no es el modelo ni
la base de datos. En este primer paso el vector store vive en memoria y se
reconstruye en cada ejecucion. Esto facilita entender el pipeline; la siguiente
iteracion puede sustituirlo por Chroma o FAISS persistente y medir la calidad
de la recuperacion.

## Estado actual
- [x] Entorno configurado (Python, Git, GitHub)
- [x] Ollama instalado con modelos de LLM y embeddings
- [x] Primera implementacion RAG con LangChain
- [ ] Implementacion RAG con LlamaIndex
- [ ] Comparacion de resultados