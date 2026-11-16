"""Verify auxiliary normal supervision without replacing masked observation learning."""
import copy
import json

import pytest
import torch

from src.config import validate_config
from src.losses import compute_losses
from src.models import build_model
from src.train import make_batch_inputs, set_seed, train_model
from src.evaluate import evaluate_model
from test_diagnostics import config_for


def new_config(tmp_path):
    c = config_for(tmp_path)
    c['experiment_id'] = 'p1_mask_x_normal'
    c['loss']['normal_reference_weight'] = 1
    return c


def test_shared_initialization_matches_p0_and_single_masked_forward(tmp_path):
    c = new_config(tmp_path)
    p0 = copy.deepcopy(c)
    p0['experiment_id'] = 'p0_mask_x'
    p0['loss'].pop('normal_reference_weight')
    set_seed(72)
    base = build_model(p0).eval()
    base_rng = torch.get_rng_state()
    set_seed(72)
    model = build_model(c).eval()
    assert torch.equal(base_rng, torch.get_rng_state())
    for k, v in base.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    calls = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: calls.append(args))
    batch = dict(x=torch.randn(2, 160, 1), valid=torch.ones(2, 160, dtype=torch.bool))
    x, visible, mask = make_batch_inputs(batch, c, torch.Generator().manual_seed(72))
    a, b = model(x, visible), base(x, visible)
    hook.remove()
    assert len(calls) == 1
    assert (x[mask] == 0).all() and torch.equal(visible, batch['valid'] & ~mask)
    for key in b:
        assert torch.equal(a[key], b[key])
    assert model.normal_reference_head[0].weight.data_ptr() != model.reconstruction_head[0].weight.data_ptr()


def test_both_targets_share_valid_selected_mask_and_gradients():
    x = torch.tensor([[[1.], [2.], [3.], [999.]]])
    x0 = torch.tensor([[[0.], [0.], [0.], [-999.]]])
    batch = dict(x=x, x0=x0, y=torch.tensor([[0, 1, 1, -1]]),
                 valid=torch.tensor([[True, True, True, False]]))
    output = dict(reconstruction=torch.zeros_like(x, requires_grad=True),
                  normal_reference=torch.ones_like(x, requires_grad=True),
                  logits=torch.zeros(1, 4, 2, requires_grad=True))
    mask = torch.tensor([[True, False, True, True]])
    loss = compute_losses(output, batch, 'x', 2, 3, mask, 'official', normal_reference_weight=4)
    assert loss['reconstruction_count'] == loss['normal_reference_count'] == 2
    assert loss['reconstruction'].item() == 5
    assert loss['normal_reference'].item() == 1
    assert loss['classification_count'] == 4
    torch.testing.assert_close(loss['total'], 2 * loss['reconstruction'] + 4 * loss['normal_reference'] + 3 * loss['classification'])
    loss['total'].backward()
    for key in ('reconstruction', 'normal_reference'):
        assert output[key].grad[0, 1].item() == output[key].grad[0, 3].item() == 0
    empty = compute_losses(output, batch, 'x', 1, 1, torch.zeros_like(mask))
    assert empty['normal_reference'].item() == empty['reconstruction'].item() == 0
    assert empty['normal_reference_count'] == empty['reconstruction_count'] == 0


def test_normal_task_reaches_encoder_and_only_its_own_head(tmp_path):
    c = new_config(tmp_path)
    model = build_model(c)
    x = torch.randn(2, 160, 1)
    valid = torch.ones(2, 160, dtype=torch.bool)
    batch = dict(x=x, x0=x * .5, y=torch.zeros(2, 160, dtype=torch.long), valid=valid)
    inputs, visible, mask = make_batch_inputs(batch, c, torch.Generator().manual_seed(72))
    output = model(inputs, visible)
    losses = compute_losses(output, batch, 'x', 1, 1, mask)
    losses['normal_reference'].backward()
    assert model.encoder.embedding_layer.weight.grad.abs().sum() > 0
    assert model.normal_reference_head[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.reconstruction_head.parameters())
    assert all(p.grad is None for p in model.anomaly_head.parameters())


def test_new_p1_training_validation_checkpoint_and_detection(tmp_path):
    from src.benchmark_eval import detection_model
    c = new_config(tmp_path)
    c['optimization']['epochs'] = 1
    set_seed(72)
    model = build_model(c)
    result = train_model(model, c, tmp_path)
    payload = torch.load(tmp_path / 'checkpoints/best.pth', weights_only=False)
    assert 'rng' in payload['training_state']
    assert any(k.startswith('normal_reference_head.') for k in payload['model_state_dict'])
    validation = payload['validation']
    assert validation['normal_reference'] > 0
    assert validation['selected_changed_points'] > 0
    assert validation['valid_total'] == pytest.approx(validation['reconstruction'] + validation['normal_reference'] + validation['valid_classification'])
    assert result['best_validation_loss'] == validation['valid_total']
    assert 'Normal Reference Loss:' in (tmp_path / 'train.log').read_text()
    model.eval()
    x, valid = torch.randn(2, 160, 1), torch.ones(2, 160, dtype=torch.bool)
    calls = []
    h = model.normal_reference_head.register_forward_hook(lambda *args: calls.append(1))
    a = detection_model(model)(x, valid)
    assert not calls
    h.remove()
    torch.testing.assert_close(a, model(x, valid)['logits'])
    restored = build_model(c)
    restored.load_state_dict(payload['model_state_dict'], strict=True)


def test_new_p1_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import test_resume
    original = test_resume.config_for
    def p1_config(path, parallel):
        c = original(path, parallel)
        c['experiment_id'] = 'p1_mask_x_normal'
        c['loss']['normal_reference_weight'] = 1
        return c
    monkeypatch.setattr(test_resume, 'config_for', p1_config)
    test_resume.test_resume_matches_uninterrupted_model_optimizer_rng_and_best(tmp_path, 'none', False)


def test_head_checkpointing_preserves_outputs_gradients_and_dropout_rng(tmp_path, monkeypatch):
    import src.models as models
    c = new_config(tmp_path)
    model = build_model(c).train()
    plain = copy.deepcopy(model)
    x = torch.randn(2, 160, 1)
    batch = dict(x=x, x0=x * .5, y=torch.zeros(2, 160, dtype=torch.long),
                 valid=torch.ones(2, 160, dtype=torch.bool))
    inputs, visible, mask = make_batch_inputs(batch, c, torch.Generator().manual_seed(72))
    set_seed(72)
    a = model(inputs, visible)
    compute_losses(a, batch, 'x', 1, 1, mask)['total'].backward()
    rng = torch.get_rng_state()
    monkeypatch.setattr(models, 'checkpoint', lambda fn, h, **kwargs: fn(h))
    set_seed(72)
    b = plain(inputs, visible)
    compute_losses(b, batch, 'x', 1, 1, mask)['total'].backward()
    assert torch.equal(rng, torch.get_rng_state())
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for p, q in zip(model.parameters(), plain.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize('weight', [0, -1, float('nan'), float('inf')])
def test_invalid_reference_weight_rejected(tmp_path, weight):
    c = new_config(tmp_path)
    c['loss']['normal_reference_weight'] = weight
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize('variant', ['p1_full_x', 'p2_full_x0'])
def test_retired_variants_cannot_silently_run_as_new_p1(tmp_path, variant):
    c = new_config(tmp_path)
    c['experiment_id'] = variant
    with pytest.raises(ValueError, match='Unsupported'):
        validate_config(c)
