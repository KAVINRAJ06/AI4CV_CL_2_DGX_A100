import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts.profile_batches import choose_batch, main


def row(batch, speed=100, peak=50, status='ok'):
    return dict(batch_size=batch, images_per_second=speed, peak_reserved_bytes=peak,
                total_memory_bytes=100, status=status)


def test_selection_uses_smallest_within_five_percent_and_strict_memory_limit():
    assert choose_batch([row(4, 94), row(8, 95), row(12, 100), row(16, 200, 85)]) == 8
    assert choose_batch([row(4, status='oom'), row(8, peak=90)]) is None
    assert choose_batch([]) is None


@pytest.mark.parametrize('failure', [None, 'oom', 'timeout', 'smoke'])
def test_process_sweep_and_recommendation(tmp_path, monkeypatch, failure):
    cfg = dict(output_dir=str(tmp_path / 'training'), device='cuda:0', batch_size=4,
               lr=.001, bottleneck_lr=.0001, model={'decoder_mode': 'spatial_fpn'},
               replay={'batch_size': 2, 'weight': 1.})
    config = tmp_path / 'cfg.yaml'
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / 'profile'
    monkeypatch.setattr(sys, 'argv', ['profile', '--config', str(config), '--output', str(output)])
    calls = []
    def fake_run(command, **kwargs):
        batch = int(command[command.index('--batch')+1])
        smoke = '--smoke' in command
        calls.append((batch, smoke))
        assert command[:4] == [sys.executable, '-u', '-m', 'scripts.profile_batches']
        assert command[command.index('--warmup')+1] == '3'
        assert command[command.index('--steps')+1] == ('3' if smoke else '10')
        if failure == 'timeout' and batch == 12:
            raise subprocess.TimeoutExpired(command, 1)
        status = 'oom' if failure == 'oom' and batch == 12 else 'error' if failure == 'smoke' and smoke else 'ok'
        directory = Path(command[command.index('--output')+1])
        (directory / 'result.json').write_text(json.dumps(row(batch, 94 if batch == 4 else 100, status=status)))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    if failure == 'smoke':
        with pytest.raises(SystemExit, match='No validated recommendation'):
            main()
        assert not (output / 'recommended.yaml').exists()
    else:
        main()
        recommended = yaml.safe_load((output / 'recommended.yaml').read_text())
        assert recommended['batch_size'] == 8
        assert recommended['lr'] == cfg['lr'] and recommended['model'] == cfg['model']
        assert recommended['replay'] == cfg['replay']
        if failure:
            assert (16, False) not in calls
    assert not Path(cfg['output_dir']).exists()
    assert yaml.safe_load(config.read_text()) == cfg
    assert calls[-1] == (8, True)


def test_existing_profile_output_is_not_overwritten(tmp_path, monkeypatch):
    config = tmp_path / 'cfg.yaml'
    config.write_text(yaml.safe_dump({'output_dir': str(tmp_path / 'training'), 'device': 'cuda'}))
    output = tmp_path / 'profile'
    output.mkdir()
    monkeypatch.setattr(sys, 'argv', ['profile', '--config', str(config), '--output', str(output)])
    with pytest.raises(FileExistsError):
        main()
