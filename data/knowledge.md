# Proyecto RAG

RAG significa Retrieval Augmented Generation. El sistema recupera fragmentos
relevantes de una coleccion de documentos y los entrega al modelo de lenguaje
como contexto para generar una respuesta fundamentada.

En este proyecto los modelos se ejecutan localmente mediante Ollama. El modelo
de embeddings transforma cada fragmento en un vector y el modelo de lenguaje
genera la respuesta final a partir de los fragmentos recuperados.