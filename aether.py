import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import asyncio
import numpy as np
from collections import deque
from datetime import datetime
from typing import Tuple

# Элитный UI
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# --- NEURAL ARCHITECTURE ---
class AetherNet(nn.Module):
    def __init__(self, input_size: int):
        super(AetherNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.SiLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- DATA & PROCESS SCOUT ---
class SystemIngestor:
    def __init__(self, buffer_size: int = 150):
        self.loss_history = deque(maxlen=buffer_size)
        
    @staticmethod
    def get_snapshot() -> np.ndarray:
        cpu = psutil.cpu_percent() / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        io = psutil.disk_io_counters()
        disk = min((io.read_bytes + io.write_bytes) / (100 * 1024 * 1024), 1.0)
        net = psutil.net_io_counters()
        network = min((net.bytes_sent + net.bytes_recv) / (10 * 1024 * 1024), 1.0)
        return np.array([cpu, ram, disk, network], dtype=np.float32)

    @staticmethod
    def get_suspect_process() -> str:
        """Находит процесс, который больше всего нагружает систему в момент аномалии."""
        try:
            procs = []
            for p in psutil.process_iter(['name', 'cpu_percent']):
                procs.append(p.info)
            top = sorted(procs, key=lambda x: x['cpu_percent'], reverse=True)[0]
            return f"{top['name']} ({top['cpu_percent']:.1f}%)"
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            return "Identifying..."

    def get_adaptive_threshold(self) -> float:
        if len(self.loss_history) < 10: return 0.1
        losses = np.array(self.loss_history)
        return losses.mean() + (losses.std() * 3)

# --- CORE ENGINE ---
class AetherEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AetherNet(input_size=4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        
    async def train_step(self, data: np.ndarray) -> float:
        self.model.train()
        tensor_data = torch.from_numpy(data).to(self.device)
        self.optimizer.zero_grad()
        output = self.model(tensor_data)
        loss = self.criterion(output, tensor_data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    async def check_anomaly(self, data: np.ndarray, threshold: float) -> Tuple[bool, float]:
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.from_numpy(data).to(self.device)
            reconstructed = self.model(tensor_data)
            loss = self.criterion(reconstructed, tensor_data).item()
            return loss > threshold, loss

# --- UI LAYOUT ---
def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    return layout

def get_status_table(metrics: np.ndarray, status: str, score: float, thresh: float, suspect: str) -> Table:
    table = Table.grid(expand=True)
    table.add_column(style="cyan", justify="right", width=20)
    table.add_column(style="white", justify="left")
    
    cpu, ram, disk, net = metrics
    table.add_row("System Status:", status)
    table.add_row("Suspect Process:", f"[bold yellow]{suspect}[/bold yellow]")
    table.add_row("Anomaly Score:", f"{score:.6f}")
    table.add_row("Threshold:", f"{thresh:.6f}")
    table.add_row("", "")
    table.add_row("CPU | RAM:", f"{cpu*100:.1f}% | {ram*100:.1f}%")
    table.add_row("Disk | Net:", f"{disk*100:.1f}% | {net*100:.1f}%")
    return table

# --- MAIN LOOP ---
async def main():
    engine = AetherEngine()
    ingestor = SystemIngestor()
    layout = make_layout()
    suspect = "None"
    
    console.print(Panel(f"Aether Engine Initializing on {engine.device}", border_style="cyan"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[dim]Building Baseline Model...", total=50)
        for _ in range(50):
            await engine.train_step(ingestor.get_snapshot())
            progress.advance(task)
            await asyncio.sleep(0.1)

    with Live(layout, refresh_per_second=4):
        layout["header"].update(Panel("[bold white]AETHER NEURAL CORE[/bold white] [dim]v1.2[/dim]", border_style="cyan", title="[bold cyan]SYSTEM ENGINE[/bold cyan]"))
        
        step = 0
        while True:
            metrics = ingestor.get_snapshot()
            thresh = ingestor.get_adaptive_threshold()
            is_anomaly, score = await engine.check_anomaly(metrics, thresh)
            
            if is_anomaly:
                suspect = ingestor.get_suspect_process()
                status = "[bold red]🚨 ANOMALY DETECTED[/bold red]"
                border = "red"
                train_msg = "[dim yellow]Training Suspended (Investigating)[/dim yellow]"
            else:
                suspect = "Stable"
                status = "[bold green]✅ SYSTEM STABLE[/bold green]"
                border = "green"
                loss = await engine.train_step(metrics)
                ingestor.loss_history.append(loss)
                train_msg = f"[dim green]Online Learning Active (Loss: {loss:.6f})[/dim green]"
            
            layout["main"].update(Panel(get_status_table(metrics, status, score, thresh, suspect), title=f"Real-time Analysis #{step}", border_style=border))
            layout["footer"].update(Panel(train_msg, border_style="cyan"))
            
            step += 1
            await asyncio.sleep(0.8)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Aether Core[/bold cyan] [white]Shutdown.[/white]")
