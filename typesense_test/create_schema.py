import typesense

client = typesense.Client(
    {
        "api_key": "xyz",
        "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
        "connection_timeout_seconds": 100,
    }
)

schema = {
    "name": "targets",
    "fields": [
        {"name": "concept_id", "type": "int32"},
        {"name": "target", "type": "string"},
        {
            "name": "embedding",
            "type": "float[]",
            "embed": {
                "from": ["target"],
                "model_config": {"model_name": "ts/multilingual-e5-small"},
                "indexing_prefix": "query:",
                "query_prefix": "query:"
            },
        },
    ],
    "default_sorting_field": "concept_id",
}


if __name__ == '__main__':

    try:
        client.collections["targets"].delete()
    except Exception as e:
        pass

    create_response = client.collections.create(schema)
    print(create_response)