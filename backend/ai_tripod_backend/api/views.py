from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from django.conf import settings

from ..chat import ChatModel
from .serializers import ChatRequestSerializer, ChatResponseSerializer


class ChatCompletionView(APIView):
    """Simple Chat Completion endpoint that delegates to ChatModel."""

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        prompt = serializer.validated_data["prompt"]
        stream = serializer.validated_data.get("stream", False)

        # instantiate or load shared ChatModel (placeholder)
        chat = ChatModel()

        if stream:
            # streaming not implemented in this scaffold
            return Response({"detail": "streaming not supported in scaffold"}, status=status.HTTP_501_NOT_IMPLEMENTED)

        output = chat.generate(prompt)
        out_ser = ChatResponseSerializer(output)
        return Response(out_ser.data)
