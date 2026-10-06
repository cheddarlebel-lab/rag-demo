# How do I limit the maximum size of an upload so large requests are rejected?

You can limit the maximum allowed payload by setting the `MAX_CONTENT_LENGTH` config key.  [1] The code example provided sets the limit to 16 megabytes. [1]

## Citations

- [1] [Uploading Files > Improving Uploads](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/#improving-uploads) (matched)
  > But how do you specify the maximum file size after which an upload is aborted? By default Flask will happily accept file uploads with an unlimited amount of memory, but you can limit that by setting the MAX_CONTENT_LENGTH config key:

Retriever: `anythingllm/flask-docs` · Analysis model: `ollama/gemma2:9b` · confidence: high · insufficient context: False

## Sources returned by the retriever

1. [Uploading Files > Improving Uploads](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/#improving-uploads) score=0.8034
2. [API > Incoming Request Data > Request > max_content_length](https://flask.palletsprojects.com/en/stable/api/#flask.Request.max_content_length) score=0.7614
3. [Configuration Handling > Builtin Configuration Values > MAX_CONTENT_LENGTH](https://flask.palletsprojects.com/en/stable/config/#MAX_CONTENT_LENGTH) score=0.7376
4. [API > Incoming Request Data > Request > max_form_memory_size](https://flask.palletsprojects.com/en/stable/api/#flask.Request.max_form_memory_size) score=0.7362
5. [Security Considerations > Resource Use](https://flask.palletsprojects.com/en/stable/web-security/#resource-use) score=0.7081
