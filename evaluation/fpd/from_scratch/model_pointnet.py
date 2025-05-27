
import torch
import numpy as np
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import LightningDataModule
from . import pointnet_cls as pointnet
from .nuscenes_fpd_dataloader import num_classes

class PointnetModule(LightningModule):
    def __init__(self, data_module: LightningDataModule = None, input_channels=4):
        super().__init__()
        self.data_module = data_module
        self.model = pointnet.Pointnet(k=num_classes, input_channels=input_channels)
        self.criterion = pointnet.get_loss()

    def forward(self, x, t, y, c, force_dropout=False):
        out = self.model(x, t, y, c, force_dropout)
        return out

    def training_step(self, batch, batch_idx):
        points, target, _ = batch

        points = points.transpose(2, 1)

        pred, trans_feat, _ = self.model(points)
        loss = self.criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct = correct.item() / float(points.size()[0])

        self.log('train/mean_correct', mean_correct)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        validation_accs = {}
        self.model.eval()
        with torch.no_grad():
            points, target, _ = batch
            points = points.transpose(2, 1)
            pred, _, _ = self.model(points)
            pred_choice = pred.data.max(1)[1]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                validation_accs[f'val/class_{cat}'] = classacc.item() / float(points[target == cat].size()[0])
                self.log(f'val/class_{cat}', validation_accs[f'val/class_{cat}'])

            correct = pred_choice.eq(target.long().data).cpu().sum()
            validation_accs['val/mean_correct'] = correct.item() / float(points.size()[0])
            self.log('val/mean_correct', validation_accs['val/mean_correct'])

        return validation_accs
    
    def test_step(self, batch, batch_idx):
        test_accs = {}
        total_preds = {}
        self.model.eval()
        with torch.no_grad():
            points, target, _ = batch
            points = points.transpose(2, 1)
            pred, _, _ = self.model(points)
            pred_choice = pred.data.max(1)[1]

            for cat in pred_choice.cpu():
                if cat.item() not in total_preds:
                    total_preds[cat.item()] = 0
                total_preds[cat.item()] += 1

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                test_accs[f'test/class_{cat}'] = classacc.item() / float(points[target == cat].size()[0])
                self.log(f'test/class_{cat}', test_accs[f'test/class_{cat}'])

            correct = pred_choice.eq(target.long().data).cpu().sum()
            test_accs['test/mean_correct'] = correct.item() / float(points.size()[0])
            self.log('test/mean_correct', test_accs['test/mean_correct'])
            
            for cat, count in total_preds.items():
                self.log(f'test/total_preds_class_{cat}', count, reduce_fx='sum')

        return test_accs
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7) 
        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 1, # after 20 epochs
        }

        return [optimizer], [scheduler]