# SSH guide for Oracle Cloud Always Free

1. Create an Ubuntu VM in Oracle Cloud Always Free
2. Download or keep your private SSH key secure
3. Connect from Windows PowerShell:

```powershell
ssh -i C:\path\to\private_key.key ubuntu@SERVER_PUBLIC_IP
```

If Oracle creates a username other than `ubuntu`, use the username shown in the instance page.

## Long runs

Use `tmux` so runs continue after closing SSH:

```bash
tmux new -s qire
python qire_ai_console.py --preset remote_discovery --profile oracle_free --time-budget-hours 24 --ram-mode throttle --ai-agent rules
```

Detach: `Ctrl+B`, then `D`

Reattach:

```bash
tmux attach -t qire
```
