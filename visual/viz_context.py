import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

import util.misc as utils


@dataclass
class VizContext:
    output_dir: Optional[Path]
    tb_writer: Optional[SummaryWriter]
    checkpoint_dir_name: str = 'train/checkpoints'
    tensorboard_dir_name: str = 'train/tensorboard'

    @classmethod
    def from_args(cls, args):
        output_dir = Path(args.output_dir) if getattr(args, 'output_dir', None) else None
        viz_ctx = cls(output_dir=output_dir, tb_writer=None)
        viz_ctx._build_output_structure()
        viz_ctx.tb_writer = viz_ctx._create_tensorboard_writer(enable_tensorboard=bool(getattr(args, 'viz', False)))
        return viz_ctx

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    @property
    def visualization_enabled(self) -> bool:
        return self.tb_writer is not None

    @property
    def should_write_artifacts(self) -> bool:
        return self.enabled and utils.is_main_process()

    @property
    def checkpoint_dir(self) -> Optional[Path]:
        if self.output_dir is None:
            return None
        return self.output_dir / self.checkpoint_dir_name

    @property
    def tensorboard_dir(self) -> Optional[Path]:
        if self.output_dir is None:
            return None
        return self.output_dir / self.tensorboard_dir_name

    def close(self):
        if self.tb_writer is None:
            return
        self.tb_writer.close()
        self.tb_writer = None

    def _build_output_structure(self):
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.checkpoint_dir
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir = self.tensorboard_dir
        if tensorboard_dir is not None:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def _create_tensorboard_writer(self, enable_tensorboard: bool):
        if not enable_tensorboard or not utils.is_main_process():
            return None
        tensorboard_dir = self.tensorboard_dir
        if tensorboard_dir is None:
            return None
        run_name = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
        log_dir = tensorboard_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(log_dir))
        logging.info('TensorBoard log dir: %s', log_dir)
        return tb_writer
