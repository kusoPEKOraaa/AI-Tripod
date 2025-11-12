from rest_framework import serializers


class ChatRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField()
    stream = serializers.BooleanField(default=False)


class ChatResponseSerializer(serializers.Serializer):
    id = serializers.CharField()
    object = serializers.CharField()
    choices = serializers.ListField()
