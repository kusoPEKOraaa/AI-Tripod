"""Script wrapper to call the training runner or management command."""
import os
import sys

if __name__ == '__main__':
    # Use Django management command to run training if Django available
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_tripod_backend.settings')
    try:
        from django.core.management import call_command

        call_command('run_train')
    except Exception:
        # fall back to direct runner
        from ai_tripod_backend.train.runner import run_exp

        run_exp()
