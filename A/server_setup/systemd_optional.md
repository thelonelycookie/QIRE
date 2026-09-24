# Optional systemd service

For v1 the recommended way is `tmux`, because it is transparent and easy to inspect.
Use systemd only if you know what you are doing.

Minimal idea:

```ini
[Unit]
Description=QIRE AI Console Remote Discovery
After=network.target

[Service]
WorkingDirectory=/home/ubuntu/qire_ai_console_v1_0
ExecStart=/home/ubuntu/qire_ai_console_v1_0/.venv/bin/python qire_ai_console.py --preset remote_discovery --profile oracle_free --time-budget-hours 24 --ram-mode throttle
Restart=no

[Install]
WantedBy=multi-user.target
```
