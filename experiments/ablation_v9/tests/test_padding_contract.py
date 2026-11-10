"""Padding ablation checks using synthetic data only."""
import copy
import json
import pickle
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.config import load_config, validate_config
from src.losses import classification_ce, compute_losses
from src.models import build_model
from src.train import make_batch_inputs, set_seed, train_model
from src.evaluate import evaluate_model
import src.train as training

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('p0_mask_x', 'p1_full_x', 'p2_full_x0')
POLICIES = ('ignore', 'official')


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config_for(variant='p0_mask_x', policy='ignore'):
    config = load_config(ROOT / 'configs' / f'{variant}.json').raw
    config['loss']['classification_padding'] = policy
    config['runtime'].update(device='cpu', parallel='none', gpu_ids=None)
    config['model'].update(d_model=32, d_proj=8, num_layers=1, num_heads=4,
                           dropout=0.0, d_ff_dropout=0.0)
    config['optimization'].update(batch_size=3, epochs=1, log_freq=1)
    return config


def synthetic_data(tmp_path):
    rows = []
    for length in (129, 157, 145, 193, 171, 163):
        x = torch.linspace(-1, 1, length).unsqueeze(-1)
        rows.append(dict(time_series=x, normal_time_series=x * 0.9,
                         labels=(torch.arange(length) % 3 == 0).long()))
    path = tmp_path / 'synthetic.pkl'
    with path.open('wb') as handle:
        pickle.dump(rows, handle)
    return path


def tensors():
    logits = torch.tensor([[[2., -1.], [-2., 1.], [-1., 3.], [3., -2.]]], requires_grad=True)
    return logits, torch.tensor([[0, 2, -1, -1]]), torch.tensor([[True, True, False, False]])


def test_official_matches_original_loss_and_gradient():
    logits, labels, valid = tensors()
    expected_logits = logits.detach().clone().requires_grad_()
    # Literal official sequence, including threshold and filtering order.
    flat_labels = (labels.view(-1) > 0.5).long()
    active = flat_labels != -1
    expected = F.cross_entropy(expected_logits.view(-1, 2)[active], flat_labels[active])
    actual, count = classification_ce(logits, labels, valid, 'official')
    assert count == labels.numel()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, expected_logits.grad)
    assert logits.grad[~valid].abs().sum() > 0


def test_ignore_removes_padding_values_and_gradients():
    logits, labels, valid = tensors()
    loss, count = classification_ce(logits, labels, valid, 'ignore')
    loss.backward()
    assert count == 2
    assert torch.count_nonzero(logits.grad[~valid]) == 0
    changed = logits.detach().clone()
    changed[~valid] = torch.tensor([-20., 20.])
    torch.testing.assert_close(classification_ce(changed, labels, valid)[0], loss)
    assert not torch.isclose(classification_ce(changed, labels, valid, 'official')[0],
                             classification_ce(logits, labels, valid, 'official')[0])


def test_without_padding_modes_are_identical():
    a, labels, valid = tensors()
    a = a[:, :2].detach().clone().requires_grad_()
    b = a.detach().clone().requires_grad_()
    first, n = classification_ce(a, labels[:, :2], valid[:, :2], 'ignore')
    second, m = classification_ce(b, labels[:, :2], valid[:, :2], 'official')
    first.backward()
    second.backward()
    assert n == m == 2
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(a.grad, b.grad)


def test_all_padding_and_empty_tensors_are_finite():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    labels = torch.full((1, 3), -1)
    valid = torch.zeros(1, 3, dtype=torch.bool)
    ignored, n = classification_ce(logits, labels, valid)
    official, m = classification_ce(logits, labels, valid, 'official')
    assert n == 0 and m == 3
    assert ignored.item() == 0
    torch.testing.assert_close(official, torch.tensor(2.).log())
    ignored.backward()
    for policy in POLICIES:
        loss, count = classification_ce(logits[:, :0], labels[:, :0], valid[:, :0], policy)
        assert count == 0 and loss.item() == 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_pair_configs_only_differ_in_padding(variant):
    first = load_config(ROOT / 'configs' / f'{variant}.json').raw
    second = load_config(ROOT / 'configs' / f'{variant}_padding_official.json').raw
    assert first['loss']['classification_padding'] == 'ignore'
    assert second['loss'].pop('classification_padding') == 'official'
    first['loss'].pop('classification_padding')
    assert first == second


@pytest.mark.parametrize('policy', [None, '', 'offical', False, [], {}])
def test_invalid_policy_fails_before_execution(policy):
    config = config_for()
    config['loss']['classification_padding'] = policy
    with pytest.raises(ValueError, match='classification_padding'):
        validate_config(config)
    logits, labels, valid = tensors()
    with pytest.raises(ValueError, match='classification_padding'):
        classification_ce(logits, labels, valid, policy)


def test_old_config_materializes_ignore_default(tmp_path):
    config = config_for()
    config['loss'].pop('classification_padding')
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    assert load_config(path).raw['loss']['classification_padding'] == 'ignore'


@pytest.mark.parametrize('variant', VARIANTS)
def test_switch_leaves_initialization_inputs_masks_and_reconstruction_unchanged(variant):
    config = config_for(variant)
    other = copy.deepcopy(config)
    other['loss']['classification_padding'] = 'official'
    set_seed(72)
    first = build_model(config)
    set_seed(72)
    second = build_model(other)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    from src.data import Sample, collate_samples
    batch = collate_samples([Sample(torch.arange(n).float().unsqueeze(-1),
                                   torch.zeros(n, 1), torch.zeros(n, dtype=torch.long), None)
                             for n in (129, 157, 145)])
    inputs = make_batch_inputs(batch, config, torch.Generator().manual_seed(72))
    other_inputs = make_batch_inputs(batch, other, torch.Generator().manual_seed(72))
    for a, b in zip(inputs, other_inputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    output = first(*inputs[:2])
    a = compute_losses(output, batch, config['input']['target'], 1, 1, inputs[2], 'ignore')
    b = compute_losses(output, batch, config['input']['target'], 1, 1, inputs[2], 'official')
    torch.testing.assert_close(a['reconstruction'], b['reconstruction'], rtol=0, atol=0)
    assert a['classification_count'] == 129 + 157 + 145
    assert b['classification_count'] == 3 * 157


@pytest.mark.parametrize('variant', VARIANTS)
def test_eval_reports_common_valid_metrics_for_same_weights(tmp_path, variant):
    config = config_for(variant)
    path = synthetic_data(tmp_path)
    set_seed(72)
    model = build_model(config)
    a = evaluate_model(model, config, str(path), [0, 1, 2])
    config['loss']['classification_padding'] = 'official'
    b = evaluate_model(model, config, str(path), [0, 1, 2])
    assert a['padding_points'] == b['padding_points'] > 0
    assert a['classification_loss_points'] == a['classification_points']
    assert b['classification_loss_points'] == b['classification_points'] + b['padding_points']
    for key in ('valid_classification', 'valid_total', 'classification_f1', 'classification_accuracy',
                'classification_precision', 'classification_recall', 'reconstruction'):
        assert a[key] == b[key]
    assert a['total'] == a['valid_total']


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('policy', POLICIES)
def test_training_evaluation_and_checkpoint_preserve_policy(tmp_path, variant, policy):
    config = config_for(variant, policy)
    config['data']['path'] = str(synthetic_data(tmp_path))
    config['split']['train_ratio'] = 0.67
    set_seed(72)
    result = train_model(build_model(config), config, tmp_path)
    payload = torch.load(tmp_path / 'checkpoints/best.pth', weights_only=False)
    assert payload['config']['loss']['classification_padding'] == policy
    assert payload['validation']['classification_padding'] == policy
    assert result['best_validation_loss'] == payload['validation']['valid_total']
    assert result['selection_metric'] == 'valid_total'
    log = (tmp_path / 'train.log').read_text()
    assert f'classification_padding={policy}' in log
    assert 'Classification points: loss=' in log


@pytest.mark.parametrize('policy', POLICIES)
def test_best_checkpoint_and_early_stopping_use_common_metric(tmp_path, monkeypatch, policy):
    config = config_for(policy=policy)
    config['data']['path'] = str(synthetic_data(tmp_path))
    config['optimization'].update(epochs=5, early_stopping_patience=1)
    sequence = iter([{'total': .01, 'valid_total': .9},
                     {'total': .2, 'valid_total': .3},
                     {'total': .4, 'valid_total': .6}])
    monkeypatch.setattr(training, 'evaluate_model', lambda *args: next(sequence))
    set_seed(72)
    result = train_model(build_model(config), config, tmp_path)
    assert result['best_epoch'] == 1
    assert result['best_validation_loss'] == .3
    assert len(list((tmp_path / 'checkpoints').glob('epoch_*.pth'))) == 3
    payload = torch.load(tmp_path / 'checkpoints/best.pth', weights_only=False)
    assert payload['epoch'] == 1
