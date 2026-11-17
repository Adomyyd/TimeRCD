"""Detailed diagnostics preserve the P1 trajectory and capture replayable updates."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_diagnostics import config_for, assert_rng_equal
from src.models import build_model
from src.models import prepare_model, unwrap_model
from src.train import train_model, set_seed, make_batch_inputs
from src.diagnostics import BatchDiagnostics, compare_states, rng_state
from src.train import restore_rng
from src.losses import compute_losses
from src.data import Sample, collate_samples


@pytest.mark.parametrize("parallel", ["none","data_parallel"])
def test_p1_detailed_diagnostics_preserve_trajectory(tmp_path,parallel):
    if parallel != "none" and torch.cuda.device_count()<2:
        pytest.skip("two GPUs required")
    config=config_for(tmp_path,parallel)
    config["experiment_id"]="p1_mask_x_normal"
    config["loss"]["normal_reference_weight"]=1
    records=[]
    for enabled in (False,True):
        raw=copy.deepcopy(config)
        if enabled:
            raw["diagnostics"]=dict(enabled=True,detailed=True,detail_interval=1,snapshot_interval=1,max_events=2,
                start_epoch=0,loss_threshold=0.00001,grad_threshold=1e20)
        run=tmp_path/str(enabled);run.mkdir()
        set_seed(72)
        model=prepare_model(build_model(raw),raw)
        train_model(model,raw,run)
        checkpoint=torch.load(run/"checkpoints/epoch_0001.pth",map_location="cpu",weights_only=False)
        records.append((checkpoint,rng_state()))
        assert all(not m._forward_hooks for m in unwrap_model(model).modules())
        del model
        if parallel!="none":torch.cuda.empty_cache()
    for key in ("model_state_dict","optimizer_state_dict","validation"):
        assert not compare_states(records[0][0][key],records[1][0][key])
    assert_rng_equal(records[0][1],records[1][1])
    directory=tmp_path/"True/diagnostics"
    rows=[json.loads(s) for s in (directory/"batches.jsonl").read_text().splitlines()]
    updates=[json.loads(s) for s in (directory/"updates.jsonl").read_text().splitlines()]
    assert len(rows)==len(updates)==6
    assert all(r["regions"]["selected"]["points"]==r["mask_count"] for r in rows)
    assert all(r["regions"]["selected"]["target_x0"]["rms"]>0 for r in rows)
    assert all(r["gradient_by_module"]["normal_reference_head"]["l2"]>0 for r in rows)
    assert all(r["sampled_activations"] for r in rows)
    expected_devices={"cpu"} if parallel=="none" else {"cuda:0","cuda:1"}
    assert set(k.split("@")[-1] for k in rows[0]["sampled_activations"])==expected_devices
    assert any(g["l2"]>0 for g in updates[0]["update_by_module"].values())
    assert (directory/"rolling_0.pth").is_file() and (directory/"rolling_1.pth").is_file()
    assert (directory/"first_event_post.pth").is_file()


def one_batch(tmp_path):
    config=config_for(tmp_path)
    config["experiment_id"]="p1_mask_x_normal"
    config["loss"]["normal_reference_weight"]=1
    config["diagnostics"]=dict(enabled=True,detailed=True,snapshot_interval=1,max_events=1,
        start_epoch=0,loss_threshold=1e-6,grad_threshold=1e20)
    set_seed(72);model=build_model(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
    x=torch.linspace(-2,2,145).unsqueeze(-1)
    batch=collate_samples([Sample(x,x*.9,torch.zeros(145,dtype=torch.long),None)]*2)
    dataset=type("Dataset",(),{"data":[{"time_series":x},{"time_series":x}]})()
    diagnostic=BatchDiagnostics(tmp_path,config);diagnostic.attach(model)
    diagnostic.begin_epoch(0,None,[0,1])
    inputs,em,mask=make_batch_inputs(batch,config,torch.Generator().manual_seed(72))
    diagnostic.before_forward();output=model(inputs,em);diagnostic.after_forward()
    loss=compute_losses(output,batch,"x",1,1,mask,classification_padding="official")
    loss["total"].backward()
    diagnostic.after_backward(model,optimizer,0,batch,inputs,em,mask,output,loss,dataset)
    optimizer.step();diagnostic.after_step(model,optimizer,0)
    return config,model,optimizer,diagnostic,batch,inputs,em,mask,output,loss,dataset


def test_saved_gradient_recreates_exact_optimizer_update_and_dropout(tmp_path):
    config,model,optimizer,diagnostic,*_=one_batch(tmp_path)
    pre=torch.load(tmp_path/"diagnostics/first_event.pth",weights_only=False)
    post=torch.load(tmp_path/"diagnostics/first_event_post.pth",weights_only=False)
    replay=build_model(config);replay.load_state_dict(pre["model_state_dict"])
    opt=torch.optim.AdamW(replay.parameters(),lr=5e-4,weight_decay=1e-5)
    opt.load_state_dict(pre["optimizer_state_dict"])
    restore_rng(pre["rng_before_forward"])
    output=replay(pre["x_input"],pre["encoder_mask"])
    losses=compute_losses(output,pre["batch"],"x",1,1,pre["mask"],classification_padding="official")
    assert losses["total"].item()==pre["losses"]["total"]
    losses["total"].backward()
    for n,p in replay.named_parameters():
        assert torch.equal(p.grad,pre["gradients"][n])
    opt.step()
    assert not compare_states(replay.state_dict(),post["model_state_dict"])
    assert not compare_states(opt.state_dict(),post["optimizer_state_dict"])
    assert_rng_equal(rng_state(),post["rng_after_step"])
    diagnostic.close()


def test_nonfinite_event_is_saved_after_finite_event_cap(tmp_path):
    config,model,opt,diag,batch,inputs,em,mask,output,loss,dataset=one_batch(tmp_path)
    first=(tmp_path/"diagnostics/first_event.pth").read_bytes()
    bad=dict(loss);bad["total"]=torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError):
        diag.after_backward(model,opt,0,batch,inputs,em,mask,output,bad,dataset)
    assert (tmp_path/"diagnostics/first_event.pth").read_bytes()==first
    assert (tmp_path/"diagnostics/event_001.pth").is_file()
    assert (tmp_path/"diagnostics/event_001_previous.pth").is_file()
    diag.close()


def test_detailed_cli_and_config_validation(monkeypatch,tmp_path):
    from run_experiment import parse_args,apply_runtime_overrides
    from src.config import load_config,validate_config
    config=load_config(Path(__file__).parents[1]/"configs/p1_mask_x_normal_padding_official.json")
    args=["run_experiment.py","--config",str(config.path),"--run-id","detailed-test","--mode","train"]
    monkeypatch.setattr("sys.argv",args+["--diagnostic-detailed"])
    with pytest.raises(SystemExit):parse_args()
    monkeypatch.setattr("sys.argv",args+["--diagnostics","--diagnostic-detailed",
        "--diagnostic-detail-interval","37","--diagnostic-snapshot-interval","250","--diagnostic-max-events","6","--epochs","25"])
    apply_runtime_overrides(config,parse_args())
    assert config.raw["diagnostics"]["detail_interval"]==37
    assert config.raw["diagnostics"]["snapshot_interval"]==250
    assert config.raw["diagnostics"]["detailed"]
    config.raw["diagnostics"]["snapshot_interval"]=-1
    with pytest.raises(ValueError):validate_config(config.raw)

@pytest.mark.skipif(torch.cuda.device_count()<2, reason="two GPUs required")
@pytest.mark.parametrize("length",[1024,10000])
def test_production_shape_p1_observation_preserves_update(tmp_path,length):
    import gc
    from src.config import load_config
    config=load_config(Path(__file__).parents[1]/"configs/p1_mask_x_normal_padding_official.json").raw
    config["runtime"].update(device="cuda",parallel="data_parallel",gpu_ids=[0,1])
    config["optimization"]["batch_size"]=32
    xs=[torch.sin(torch.arange(length,dtype=torch.float32)/(8+i)).unsqueeze(-1) for i in range(32)]
    batch=collate_samples([Sample(x,x*.9,(torch.arange(length)%11==0).long(),None) for x in xs])
    dataset=type("Dataset",(),{"data":[{"time_series":x} for x in xs]})()
    inputs,em,mask=make_batch_inputs(batch,config,torch.Generator().manual_seed(72))
    results=[]
    for enabled in [False,True]:
        raw=copy.deepcopy(config)
        if enabled:
            raw["diagnostics"]=dict(enabled=True,detailed=True,snapshot_interval=0,max_events=1,
                                    start_epoch=0,loss_threshold=1e20,grad_threshold=1e20)
        set_seed(72);model=prepare_model(build_model(raw),raw)
        opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
        dev=torch.device("cuda:0")
        bd={k:v.to(dev) for k,v in batch.items() if isinstance(v,torch.Tensor)}
        observer=BatchDiagnostics(tmp_path/"production",raw) if enabled else None
        if observer:
            observer.attach(model);observer.begin_epoch(0,None,list(range(32)));observer.before_forward()
        output=model(inputs.to(dev),em.to(dev))
        if observer:observer.after_forward()
        loss=compute_losses(output,bd,"x",1,1,mask.to(dev),classification_padding="official")
        loss["total"].backward()
        if observer:observer.after_backward(model,opt,0,batch,inputs,em,mask,output,loss,dataset)
        opt.step()
        if observer:observer.after_step(model,opt,0);observer.close()
        from src.diagnostics import cpu_copy
        results.append((cpu_copy(unwrap_model(model).state_dict()),cpu_copy(opt.state_dict()),rng_state()))
        del output,loss,bd,opt,model
        gc.collect();torch.cuda.empty_cache()
    assert not compare_states(results[0][0],results[1][0])
    assert not compare_states(results[0][1],results[1][1])
    assert_rng_equal(results[0][2],results[1][2])

@pytest.mark.parametrize("parallel", ["none", "data_parallel"])
def test_interval_diagnostics_preserve_updates_and_rng(tmp_path, parallel):
    if parallel != "none" and torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    config = config_for(tmp_path, parallel)
    config["experiment_id"] = "p1_mask_x_normal"
    config["loss"]["normal_reference_weight"] = 1
    results = []
    for enabled in (False, True):
        raw = copy.deepcopy(config)
        if enabled:
            raw["diagnostics"] = dict(enabled=True, detailed=True, detail_interval=2,
                snapshot_interval=0, max_events=2, start_epoch=0,
                loss_threshold=1e20, grad_threshold=1e20)
        run = tmp_path / ("interval" if enabled else "plain")
        run.mkdir()
        set_seed(72)
        model = prepare_model(build_model(raw), raw)
        train_model(model, raw, run)
        results.append((torch.load(run / "checkpoints/epoch_0001.pth",
                                   map_location="cpu", weights_only=False), rng_state()))
        del model
        if parallel != "none":
            torch.cuda.empty_cache()
    for key in ("model_state_dict", "optimizer_state_dict", "validation"):
        assert not compare_states(results[0][0][key], results[1][0][key])
    assert_rng_equal(results[0][1], results[1][1])
    directory = tmp_path / "interval/diagnostics"
    rows = [json.loads(s) for s in (directory / "batches.jsonl").read_text().splitlines()]
    updates = [json.loads(s) for s in (directory / "updates.jsonl").read_text().splitlines()]
    assert len(rows) == len(updates) == 6
    assert [r["observation_level"] for r in rows] == ["detailed", "light", "detailed"] * 2
    for r, u in zip(rows, updates):
        assert r["batch"] == u["batch"] and r["epoch"] == u["epoch"]
        if r["observation_level"] == "light":
            assert "state_before" not in r and "state_after" not in u
            assert "update_by_module" not in u
            assert r["gradient_by_parameter"] and r["gradient_l2"] > 0
            assert r["tensor_stats"]["hidden"]["rms"] > 0
            assert u["nonfinite_state"] == []
            assert all(k.startswith("anomaly_head.1@") for k in r["sampled_activations"])


def test_unscheduled_event_promotes_and_repeated_threshold_stays_light(tmp_path):
    config, model, opt, diag, batch, inputs, em, mask, output, loss, dataset = one_batch(tmp_path)
    detail = diag.detail
    detail.interval = 0
    detail.detail_interval = 100
    detail.max_events = 2
    detail.sampled = False
    diag.sample_batches = [[0, 1]] * 4
    # First event already exists. A tenfold escalation on an unscheduled batch saves detail.
    loss = dict(loss)
    loss["total"] = loss["total"] * 100
    diag.after_backward(model, opt, 1, batch, inputs, em, mask, output, loss, dataset)
    assert detail.before_parameters is not None
    opt.step()
    diag.after_step(model, opt, 1)
    assert (tmp_path / "diagnostics/event_001_post.pth").is_file()
    # The same persistent finite threshold must not make every later batch expensive.
    output = dict(output)
    output["h"] = torch.full_like(output["h"], 2000)
    diag.after_backward(model, opt, 2, batch, inputs, em, mask, output, loss, dataset)
    assert detail.before_parameters is None
    opt.step()
    diag.after_step(model, opt, 2)
    rows = [json.loads(s) for s in (tmp_path / "diagnostics/batches.jsonl").read_text().splitlines()]
    assert rows[-2]["observation_level"] == "detailed"
    assert rows[-1]["observation_level"] == "light"
    assert "hidden_scale" in rows[-1]["triggers"]
    assert rows[-1]["hidden_moments"]["rms"] == pytest.approx(2000)
    # Nonfinite updates must still be caught on an unscheduled batch.
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("inf")
    with pytest.raises(FloatingPointError, match="Nonfinite optimizer update"):
        diag.after_step(model, opt, 3)
    assert (tmp_path / "diagnostics/nonfinite_update_post.pth").is_file()
    diag.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compact_reductions_match_reference_without_rng_changes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from src.diagnostics import tensor_l2_values, compact_tensor_stats
    x = torch.tensor([-1e20, 0, 2e20], device=device, dtype=torch.float32)
    before = rng_state()
    norms = tensor_l2_values([("x", x)])
    stats = compact_tensor_stats(x)
    assert norms["x"] == pytest.approx(torch.linalg.vector_norm(x.cpu().double()).item(), rel=1e-12)
    assert stats["rms"] == pytest.approx(norms["x"] / np.sqrt(3), rel=1e-12)
    assert stats["min"] == x.cpu().min().item() and stats["nonfinite"] == 0
    x[1] = float("nan")
    assert np.isnan(tensor_l2_values([("x", x)])["x"])
    assert compact_tensor_stats(x)["nonfinite"] == 1
    assert_rng_equal(before, rng_state())
