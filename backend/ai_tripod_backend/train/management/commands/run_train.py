from django.core.management.base import BaseCommand

from ...train.runner import run_exp


class Command(BaseCommand):
    help = 'Run training job (scaffold).'

    def add_arguments(self, parser):
        parser.add_argument('--config', type=str, help='Path to training config (optional)')

    def handle(self, *args, **options):
        cfg = {'config': options.get('config')}
        run_exp(cfg)
        self.stdout.write(self.style.SUCCESS('run_train completed (scaffold)'))
