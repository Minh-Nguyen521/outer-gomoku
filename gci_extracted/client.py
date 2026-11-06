"""
Gomoku online, bản dành cho bên vào phòng: chỉ có Client và Xem log. Python >= 3.9, không cần cài thêm thư viện.

    python client.py    # mở app

Cần 2 file đặt cùng thư mục: client.py và online_wrapper_template.py.
1. Sửa BOT_NAME, load_bot và choose_move trong online_wrapper_template.py (hướng dẫn nằm ở đầu file đó).
2. Mở app -> Client -> chọn file bot, chờ bot tải xong thì nút "Vào phòng" mới mở.
3. Nhập host và match code do server gửi, rồi bấm "Vào phòng". Mọi thứ sau đó chạy tự động:
   ván tự bắt đầu sau khi đếm ngược, mỗi lượt nhiều ván (mặc định 4), hai bên đổi màu sau mỗi ván, không có xin thua.
4. Mỗi ván được lưu log vào log_games/ cạnh file này, xem lại bằng mục "Xem log".

Luật và đồng hồ do server quyết định: Outer-Open Free-style (nước 1 nằm trong 2 vòng ngoài, từ 5 quân liên tiếp trở lên là
thắng). Đồng hồ chạy từ lúc tới lượt tới lúc server NHẬN được nước đi, nên độ trễ mạng được tính vào giờ của bạn.
Quá giờ là thua, kể cả khi đang mất kết nối. Ô là board[x][y], x = hàng, y = cột, tính từ 0.
"""
from __future__ import annotations

import importlib.util
import json
import queue
import re
import secrets
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # máy không có tkinter vẫn chạy được bot bằng online_wrapper_template.py
    tk = None

# =====================================================================
# HẰNG SỐ VÀ HÀM CHUNG
# =====================================================================

BOARD_SIZE = 15
OUTER_RING_WIDTH = 2
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
BLACK, WHITE = 1, 2

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = BASE_DIR / "log_games"
LONG_POLL_WAIT = 8.0  # phải nhỏ hơn OFFLINE_AFTER: client đang long-poll vẫn được tính là online

# Trạng thái phòng
WAITING = "waiting"    # chờ đối thủ vào phòng
LOADING = "loading"    # đối thủ đã vào phòng, chờ họ gửi ready
READY = "ready"        # đủ 2 bên đã sẵn sàng, đang đếm ngược tới ván đầu của lượt
PLAYING = "playing"
FINISHED = "finished"
CLOSED = "closed"      # server đã đóng phòng


def is_outer(x: int, y: int) -> bool:
    """Ô thuộc 2 vòng ngoài, nơi được phép đặt nước 1."""
    w = OUTER_RING_WIDTH
    return x < w or y < w or x >= BOARD_SIZE - w or y >= BOARD_SIZE - w


def find_winning_line(board: list[list[int]], x: int, y: int, player: int) -> list[tuple[int, int]]:
    """Các ô của hàng từ 5 quân trở lên đi qua (x, y), đã sắp theo thứ tự trên hàng; không có thì trả []."""
    for dx, dy in DIRECTIONS:
        line = [(x, y)]
        for step in (-1, 1):
            nx, ny = x + dx * step, y + dy * step
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
                line.append((nx, ny))
                nx += dx * step
                ny += dy * step
        if len(line) >= 5:
            return sorted(line)
    return []


def save_game_log(state: dict[str, Any], log_dir: Path | None) -> str | None:
    """Lưu ván đã kết thúc từ một state, theo định dạng của GUI_Checklog. Trả về tên file, lỗi thì trả None."""
    if log_dir is None or not state.get("result"):
        return None
    result, series = state["result"], state["series"]
    names = {stone: state["players"][str(stone)]["name"] for stone in (BLACK, WHITE)}
    data = {
        "match": "online",
        "round": series["round"],
        "round_game": series["game"],
        "score": {
            names[BLACK]: state["players"][str(BLACK)]["wins"],
            names[WHITE]: state["players"][str(WHITE)]["wins"],
            "draws": series["draws"],
        },
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "rules": state["rules"],
        "players": {"black": names[BLACK], "white": names[WHITE]},
        "winner": names.get(result["winner"]),
        "result": "draw" if result["winner"] == 0 else "win",
        "reason": result["text"],
        "reason_code": result["reason"],
        "moves": state["moves"],
        "final_state": state["board"],
    }
    log_dir = Path(log_dir)
    path = log_dir / f"online_{datetime.now():%Y%m%d_%H%M%S}_r{series['round']}_g{series['game']}.json"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        traceback.print_exc()
        return None
    return path.name


def _clean_name(name: Any, default: str) -> str:
    text = " ".join(str(name or "").split())[:32]
    return text or default


# =====================================================================
# KẾT NỐI TỚI SERVER
# =====================================================================
class ApiError(Exception):
    """Server đã trả lời và từ chối yêu cầu."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.message, self.status = message, status


class ConnectionLost(Exception):
    """Không nhận được câu trả lời hợp lệ từ server."""


def normalize_url(url: str) -> str:
    """Nhận '100.1.2.3:8765', 'localhost:8765', 'abc.ngrok-free.app' hoặc URL đầy đủ."""
    text = url.strip().rstrip("/")
    if not text:
        raise ValueError("Chưa nhập địa chỉ host.")
    if "://" in text:
        return text
    host = text.split(":")[0]
    is_direct = ":" in text or host == "localhost" or host.replace(".", "").isdigit()
    return f"{'http' if is_direct else 'https'}://{text}"


class GomokuClient:
    def __init__(self, url: str, token: str | None = None, timeout: float = 10.0) -> None:
        self.url, self.token, self.timeout = normalize_url(url), token, timeout
        # http:// là kết nối trực tiếp (localhost, Tailscale, LAN), không đi qua proxy hệ thống.
        handlers = [urllib.request.ProxyHandler({})] if self.url.startswith("http://") else []
        self._opener = urllib.request.build_opener(*handlers)

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float | None = None) -> dict[str, Any]:
        headers = {"User-Agent": "gomoku-online/1", "ngrok-skip-browser-warning": "1"}
        if self.token:
            headers["X-Token"] = self.token
        data = None
        if method == "POST":
            data = json.dumps(body or {}).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read().decode("utf-8"))["error"]
            except (ValueError, KeyError, TypeError, OSError):
                # Không phải JSON của server này, ví dụ trang lỗi của ngrok khi tunnel đã tắt.
                raise ConnectionLost(f"HTTP {error.code} từ {self.url}") from error
            raise ApiError(message, error.code) from error
        except ValueError as error:
            raise ConnectionLost(f"{self.url} không trả về JSON (có đúng địa chỉ host không?)") from error
        except OSError as error:
            raise ConnectionLost(f"Không kết nối được {self.url}: {error}") from error

    def info(self) -> dict[str, Any]:
        return self._call("GET", "/api/info")

    def join(self, code: str, name: str) -> dict[str, Any]:
        reply = self._call("POST", "/api/join", {"code": code, "name": name})
        self.token = reply["token"]
        return reply["state"]

    def state(self, since: int | None = None, wait: float = 0.0) -> dict[str, Any]:
        query = f"?since={since}&wait={wait}" if since is not None else ""
        return self._call("GET", "/api/state" + query, timeout=self.timeout + wait)

    def move(self, x: int, y: int, ply: int, game_no: int | None = None) -> dict[str, Any]:
        return self._call("POST", "/api/move", {"x": x, "y": y, "ply": ply, "game_no": game_no})

    def ready(self) -> dict[str, Any]:
        return self._call("POST", "/api/ready")

    def leave(self) -> dict[str, Any]:
        return self._call("POST", "/api/leave")


class BotRoom:
    """Tham số `room` của load_bot. load_bot chạy TRƯỚC khi vào phòng, nên các lệnh này chỉ có tác dụng ở phía bot."""

    def __init__(self, on_status: Callable[[str], Any], on_ready: Callable[[], Any]) -> None:
        self._on_status, self._on_ready = on_status, on_ready
        self.is_ready = False

    def status(self, text: str) -> None:
        """Hiện tiến độ tải bot."""
        self._on_status(text)

    def ready(self) -> None:
        """Báo bot đã sẵn sàng: từ lúc này mới được vào phòng."""
        if not self.is_ready:
            self.is_ready = True
            self._on_ready()


def send_ready(client: GomokuClient) -> None:
    """Gửi ready ngay sau khi vào phòng, tự thử lại nếu mất mạng."""
    for attempt in range(30):
        try:
            client.ready()
            return
        except ConnectionLost:
            if attempt == 29:
                raise
            time.sleep(1.0)


def build_turn_info(state: dict[str, Any], margin: float = 0.5) -> dict[str, Any]:
    """Tham số `info` truyền vào choose_move của bot."""
    time_left = state["time_left"]
    return {
        "ply": state["ply"],
        "game_no": state["game_no"],
        "round_game": state["series"]["game"],
        "games": state["series"]["games"],
        "last_move": state["last_move"],
        "moves": state["moves"],
        "rules": state["rules"],
        "total_left": state["players"][str(state["you"])]["total_left"],
        "move_left": state["move_left"],
        "time_left": time_left,
        "time_budget": None if time_left is None else max(0.0, time_left - margin),
        "opponent": state["players"][str(3 - state["you"])]["name"],
    }


def _play_turn(client, state, bot_move, margin, thinking, log) -> bool:
    info = build_turn_info(state, margin)
    thinking.set()
    try:
        x, y = bot_move(state["board"], state["you"], info)
        x, y = int(x), int(y)
    except Exception:
        log("choose_move bị lỗi:\n" + traceback.format_exc())
        return False
    finally:
        thinking.clear()

    for _ in range(5):
        try:
            client.move(x, y, state["ply"], state["game_no"])
            log(f"Nước {state['ply'] + 1}: ({x}, {y})")
            return True
        except ConnectionLost as error:  # nếu lần trước đã tới server, lần gửi lại bị 409 và ta đồng bộ lại
            log(f"Gửi nước đi thất bại, thử lại: {error}")
            time.sleep(0.5)
        except ApiError as error:
            log(f"Nước ({x}, {y}) bị từ chối: {error.message}")
            return False
    return False


def run_bot(
    url: str,
    code: str,
    name: str,
    bot_move: Callable[[list[list[int]], int, dict[str, Any]], tuple[int, int]],
    margin: float = 0.5,
    once: bool = False,
    log: Callable[[str], None] = print,
    log_dir: Path | None = DEFAULT_LOG_DIR,
    load_bot: Callable[[BotRoom], Any] | None = None,
) -> None:
    """Vào phòng rồi chơi tới khi phòng đóng (hoặc hết lượt hiện tại nếu once=True), không cần giao diện."""
    if log is print and hasattr(sys.stdout, "reconfigure"):
        # stdout bị chuyển hướng trên Windows dùng codepage cũ (cp950, cp1252): in tiếng Việt sẽ làm bot crash giữa ván.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = GomokuClient(url)
    if load_bot is not None:  # tải xong mới vào phòng; lỗi thì dừng luôn, chưa kết nối gì cả
        log("Đang tải bot...")
        room = BotRoom(log, lambda: log("Bot đã sẵn sàng."))
        load_bot(room)
        room.ready()
    client.join(code, name)
    log(f"Đã vào phòng tại {client.url}.")

    stop, thinking = threading.Event(), threading.Event()

    def heartbeat() -> None:  # lúc đang nghĩ không có long-poll, cần báo để không bị coi là offline
        while not stop.wait(5.0):
            if thinking.is_set():
                try:
                    client.state()
                except (ConnectionLost, ApiError):
                    pass

    threading.Thread(target=heartbeat, daemon=True).start()
    since, last_status, announced = -1, None, 0
    try:
        send_ready(client)
        log("Ván sẽ tự bắt đầu.")
        while True:
            try:
                state = client.state(since, LONG_POLL_WAIT)
            except ConnectionLost as error:
                log(f"Mất kết nối, thử lại: {error}")
                time.sleep(2.0)
                continue
            since = state["version"]
            if state["status"] != last_status:
                last_status = state["status"]
                series = state["series"]
                log(f"Trạng thái: {last_status} (lượt {series['round']}, ván {series['game']}/{series['games']}, "
                    f"bạn cầm {'Đen' if state['you'] == BLACK else 'Trắng'})")
            if state["result"] and announced != state["game_no"]:
                announced = state["game_no"]
                mine, theirs = (state["players"][str(stone)]["wins"] for stone in (state["you"], 3 - state["you"]))
                log(f"Kết thúc ván: {state['result']['text']} Điểm: bạn {mine}, đối thủ {theirs}, hoà {state['series']['draws']}.")
                saved = save_game_log(state, log_dir)
                if saved:
                    log(f"Đã lưu log: {saved}")
                if once and state["series"]["over"]:
                    return
            if state["status"] == CLOSED:
                log("Server đã đóng phòng.")
                return
            if state["your_turn"] and not _play_turn(client, state, bot_move, margin, thinking, log):
                since = -1  # nước đi bị từ chối hoặc lỗi: lấy lại state rồi thử lại
                time.sleep(0.5)
    finally:
        stop.set()


# =====================================================================
# FILE BOT
# =====================================================================
DEFAULT_WRAPPER = BASE_DIR / "online_wrapper_template.py"
BOT_MARGIN = 0.5  # giây chừa cho mạng và Python


def read_bot_name(path: Path) -> str | None:
    """Đọc BOT_NAME trong file wrapper mà không chạy file đó."""
    try:
        found = re.search(r"""^BOT_NAME\s*=\s*["'](.+?)["']""", Path(path).read_text(encoding="utf-8"), re.MULTILINE)
    except (OSError, UnicodeDecodeError):
        return None
    return found.group(1) if found else None


class WrapperBot:
    """Gọi choose_move trong một file wrapper. Nạp lại mỗi lần vào phòng để nhận thay đổi mới nhất của file."""

    def __init__(self, path: Path) -> None:
        for folder in (str(BASE_DIR), str(path.parent)):  # file wrapper import gomoku_online và các module đặt cạnh nó
            if folder not in sys.path:
                sys.path.insert(0, folder)
        spec = importlib.util.spec_from_file_location(f"gomoku_wrapper_{secrets.token_hex(4)}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"{path} không phải file Python.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.choose_move = getattr(module, "choose_move", None)
        if not callable(self.choose_move):
            raise ValueError(f"{path.name} chưa có hàm choose_move.")
        self.load_bot = getattr(module, "load_bot", None)

    def prepare(self, room: BotRoom) -> None:
        """Chạy load_bot của file wrapper (nếu có)."""
        if callable(self.load_bot):
            self.load_bot(room)

    def choose(self, state: dict[str, Any], age: float) -> tuple[int, int]:
        info = build_turn_info(state, BOT_MARGIN)
        if info["time_budget"] is not None:
            info["time_budget"] = max(0.0, info["time_budget"] - age)
        x, y = self.choose_move(state["board"], state["you"], info)
        return int(x), int(y)


# =====================================================================
# GUI: DÙNG CHUNG
# =====================================================================
CELL = 38
MARGIN = 32
BOARD_EXTENT = MARGIN * 2 + CELL * (BOARD_SIZE - 1)
BOARD_BG = "#deb979"
STONE_COLORS = {BLACK: "#202020", WHITE: "#f8f8f8"}
STONE_OUTLINES = {BLACK: "#111111", WHITE: "#777777"}
PAGE_BG = "#e8edf2"
PANEL_BG = "#f8fafc"
NAVY = "#1f3b57"
ACTIVE_BG = "#dbeafe"
LOW_TIME_SECONDS = 10.0

STATUS_TEXT = {
    WAITING: "ĐANG CHỜ ĐỐI THỦ",
    LOADING: "CHỜ BOT CLIENT SẴN SÀNG",
    READY: "ĐÃ ĐỦ 2 BOT",
    PLAYING: "ĐANG CHƠI",
    FINISHED: "KẾT THÚC VÁN",
    CLOSED: "PHÒNG ĐÃ ĐÓNG",
}
STATUS_COLOR = {WAITING: "#b45309", LOADING: "#b45309", READY: "#1d4ed8", PLAYING: "#15803d", FINISHED: "#7c2d12", CLOSED: "#64748b"}


def fmt_clock(seconds: float | None) -> str:
    if seconds is None:
        return "∞"
    seconds = max(0.0, seconds)
    if seconds < LOW_TIME_SECONDS:
        return f"{seconds:.1f}"
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


# =====================================================================
# GUI: XEM LOG
# =====================================================================
@dataclass
class GameLog:
    path: Path
    data: dict[str, Any]
    moves: list[dict[str, Any]]
    boards: list[list[list[int]]]
    winning_line: list[tuple[int, int]]


def _read_log_board(value: Any) -> list[list[int]]:
    if (
        not isinstance(value, list)
        or len(value) != BOARD_SIZE
        or any(not isinstance(row, list) or len(row) != BOARD_SIZE for row in value)
    ):
        raise ValueError("Board state must be a 15 x 15 matrix.")
    if any(cell not in (0, 1, 2) for row in value for cell in row):
        raise ValueError("Board may only contain the values 0, 1 or 2.")
    return [row.copy() for row in value]


def _log_move_coordinates(move: dict[str, Any]) -> tuple[int, int]:
    action = move.get("action", move)
    if isinstance(action, dict):
        x, y = action.get("x"), action.get("y")
    elif isinstance(action, (list, tuple)) and len(action) == 2:
        x, y = action
    else:
        raise ValueError("Move has no valid action {x, y}.")
    if (
        not isinstance(x, int)
        or isinstance(x, bool)
        or not isinstance(y, int)
        or isinstance(y, bool)
        or not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE)
    ):
        raise ValueError(f"Invalid move coordinates: {x}, {y}.")
    return x, y


def load_game_log(path: Path) -> GameLog:
    try:
        with path.open("r", encoding="utf-8") as log_file:
            data = json.load(log_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read log '{path}': {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("moves"), list):
        raise ValueError("Invalid log format: expected a JSON object with a moves list.")

    moves = data["moves"]
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    boards = [[row.copy() for row in board]]
    winning_line: list[tuple[int, int]] = []
    for index, move in enumerate(moves, start=1):
        if not isinstance(move, dict):
            raise ValueError(f"Move {index} is not a JSON object.")
        x, y = _log_move_coordinates(move)
        player = move.get("player")
        if player not in (BLACK, WHITE):
            raise ValueError(f"Move {index} has no valid player (1 or 2).")
        if board[x][y] != 0:
            raise ValueError(f"Move {index} plays on an occupied cell ({x}, {y}).")
        state = move.get("state")
        if state is not None and _read_log_board(state) != board:
            raise ValueError(f"State of move {index} does not match the move history.")
        board[x][y] = player
        boards.append([row.copy() for row in board])
        winning_line = find_winning_line(board, x, y, player) or winning_line
    if "final_state" in data:
        _read_log_board(data["final_state"])
    return GameLog(path, data, moves, boards, winning_line)


class LogViewer:
    """Duyệt và phát lại các ván trong một folder log."""

    def __init__(self, root: tk.Misc, log_dir: Path) -> None:
        self.root, self.log_dir = root, Path(log_dir)
        self.root.title("Gomoku - Game Log Viewer")
        self.root.minsize(1000, 700)
        self.root.configure(bg=PAGE_BG)
        self.game_files: list[Path] = []
        self.game_index = -1
        self.game: GameLog | None = None
        self.current_ply = 0
        self.cell_size = 36
        self.margin = 30
        self._build_ui()
        self._install_keyboard_shortcuts()
        self._refresh_game_files()
        if self.game_files:
            self._load_game_at(0)
        else:
            self._set_status(f"Chưa có log trong {self.log_dir}.")

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=NAVY, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="GOMOKU GAME LOG", bg=NAVY, fg="white", font=("Segoe UI", 18, "bold")).pack(side="left")
        self.game_title = tk.Label(header, text="No game loaded", bg=NAVY, fg="#dce8f2", font=("Segoe UI", 10))
        self.game_title.pack(side="right")

        game_navigation = tk.Frame(self.root, bg=PAGE_BG, pady=8)
        game_navigation.pack(fill="x")
        game_controls = tk.Frame(game_navigation, bg=PAGE_BG)
        game_controls.pack(anchor="center")
        ttk.Button(game_controls, text="◀ Prev Game", command=self._previous_game).pack(side="left", padx=6)
        self.game_counter = tk.Label(game_controls, text="0 / 0", bg=PAGE_BG, fg=NAVY, font=("Segoe UI", 11, "bold"), width=12)
        self.game_counter.pack(side="left", padx=6)
        ttk.Button(game_controls, text="Next Game ▶", command=self._next_game).pack(side="left", padx=6)

        footer = tk.Label(self.root, text="Enter: Next Game    |    Space: Next Move", bg=PAGE_BG, fg="#64748b", font=("Segoe UI", 9), pady=4)
        footer.pack(side="bottom")
        self.status = tk.Label(self.root, text="", anchor="w", bg="#dbe4ec", fg="#334155", padx=10, pady=5, font=("Segoe UI", 9))
        self.status.pack(fill="x", side="bottom")

        body = tk.Frame(self.root, bg=PAGE_BG, padx=14, pady=12)
        body.pack(fill="both", expand=True)
        board_panel = tk.Frame(body, bg=PAGE_BG)
        board_panel.pack(side="left", fill="both", expand=True)
        extent = self.margin * 2 + self.cell_size * (BOARD_SIZE - 1)
        self.canvas = tk.Canvas(board_panel, width=extent, height=extent, bg=BOARD_BG, highlightthickness=1, highlightbackground="#9d7845")
        self.canvas.pack(pady=(0, 10))
        self.result_label = tk.Label(board_panel, text="", bg=PAGE_BG, fg=NAVY, font=("Segoe UI", 11, "bold"), wraplength=extent)
        self.result_label.pack()
        self.ply_label = tk.Label(board_panel, text="Ply: 0 / 0", bg=PAGE_BG, fg="#334155", font=("Segoe UI", 10))
        self.ply_label.pack(pady=(3, 0))
        move_controls = tk.Frame(board_panel, bg=PAGE_BG)
        move_controls.pack(anchor="center", pady=(10, 0))
        ttk.Button(move_controls, text="◀ Prev Move", command=self._previous_move).pack(side="left", padx=6)
        ttk.Button(move_controls, text="Next Move ▶", command=self._next_move).pack(side="left", padx=6)

        side_panel = tk.Frame(body, bg=PANEL_BG, padx=12, pady=12, width=350)
        side_panel.pack(side="right", fill="y", padx=(14, 0))
        side_panel.pack_propagate(False)
        tk.Label(side_panel, text="MOVE HISTORY", bg=PANEL_BG, fg=NAVY, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.move_list = tk.Listbox(side_panel, font=("Consolas", 10), activestyle="none", selectmode=tk.SINGLE, bg="white", fg="#1e293b", relief="solid", borderwidth=1)
        self.move_list.pack(fill="both", expand=True, pady=(8, 10))
        self.move_list.bind("<<ListboxSelect>>", self._on_move_selected)
        ttk.Button(side_panel, text="Load Log File...", command=self._choose_file).pack(fill="x", pady=(8, 2))
        self._draw_board([[0] * BOARD_SIZE for _ in range(BOARD_SIZE)], [])

    def _install_keyboard_shortcuts(self) -> None:
        shortcut_tag = "GomokuLogShortcuts"
        self.root.bind_class(shortcut_tag, "<Return>", self._on_enter)
        self.root.bind_class(shortcut_tag, "<space>", self._on_space)

        def add_shortcut_tag(widget: tk.Misc) -> None:
            tags = list(widget.bindtags())
            tags.insert(1, shortcut_tag)
            widget.bindtags(tuple(tags))
            for child in widget.winfo_children():
                add_shortcut_tag(child)

        add_shortcut_tag(self.root)

    def _refresh_game_files(self) -> None:
        current_path = self.game.path if self.game else None
        files = sorted(
            (path for path in self.log_dir.glob("*.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if self.log_dir.exists() else []
        if current_path and current_path not in files:
            files.append(current_path)
        self.game_files = files
        if current_path in self.game_files:
            self.game_index = self.game_files.index(current_path)

    def _choose_file(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Select a Gomoku game log",
            initialdir=str(self.log_dir if self.log_dir.exists() else BASE_DIR),
            filetypes=(("JSON log files", "*.json"), ("All files", "*.*")),
        )
        if filename:
            self._load_path(Path(filename))

    def _load_path(self, path: Path) -> None:
        try:
            game = load_game_log(path)
        except ValueError as error:
            messagebox.showerror("Cannot load log", str(error), parent=self.root)
            self._set_status(str(error))
            return
        self._refresh_game_files()
        try:
            self.game_index = self.game_files.index(path)
        except ValueError:
            self.game_files.insert(0, path)
            self.game_index = 0
        self.game = game
        self.current_ply = len(game.moves)
        self._populate_move_list()
        self._render_current_position()
        self.game_title.configure(text=path.name)
        total = len(self.game_files)
        self.game_counter.configure(text=f"{self.game_index + 1} / {total}")
        self._set_status(f"Loaded {path}")

    def _load_game_at(self, index: int) -> None:
        if not self.game_files:
            self._set_status(f"Chưa có log trong {self.log_dir}.")
            return
        self.game_index = index % len(self.game_files)
        self._load_path(self.game_files[self.game_index])

    def _populate_move_list(self) -> None:
        self.move_list.delete(0, tk.END)
        if self.game is None:
            return
        players = self.game.data.get("players", {})
        for index, move in enumerate(self.game.moves, start=1):
            player = move.get("player")
            color = "Black" if player == BLACK else "White"
            bot = move.get("bot") or players.get("black" if player == BLACK else "white") or color
            x, y = _log_move_coordinates(move)
            self.move_list.insert(tk.END, f"{index:03d}  {color:<5} {bot[:16]:<16} ({x:2d}, {y:2d})")

    def _render_current_position(self) -> None:
        if self.game is None:
            return
        board = self.game.boards[self.current_ply]
        winning_line = self.game.winning_line if self.current_ply == len(self.game.moves) else []
        self._draw_board(board, winning_line)
        self.ply_label.configure(text=f"Ply: {self.current_ply} / {len(self.game.moves)}")
        self._update_result_label()
        self.move_list.selection_clear(0, tk.END)
        if self.current_ply:
            index = self.current_ply - 1
            self.move_list.selection_set(index)
            self.move_list.activate(index)
            self.move_list.see(index)

    def _draw_board(self, board: list[list[int]], winning_line: list[tuple[int, int]]) -> None:
        self.canvas.delete("all")
        size, margin = self.cell_size, self.margin
        extent = margin + size * (BOARD_SIZE - 1)
        for index in range(BOARD_SIZE):
            position = margin + index * size
            self.canvas.create_line(margin, position, extent, position, fill="#624b2c")
            self.canvas.create_line(position, margin, position, extent, fill="#624b2c")
            self.canvas.create_text(position, margin - 21, text=str(index), fill="#4b3b26", font=("Segoe UI", 8))
            self.canvas.create_text(margin - 21, position, text=str(index), fill="#4b3b26", font=("Segoe UI", 8))
        winning_cells = set(winning_line)
        radius = size * 0.38
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                player = board[x][y]
                if not player:
                    continue
                cx, cy = margin + y * size, margin + x * size
                won = (x, y) in winning_cells
                self.canvas.create_oval(
                    cx - radius, cy - radius, cx + radius, cy + radius,
                    fill=STONE_COLORS[player],
                    outline="#e11d48" if won else STONE_OUTLINES[player],
                    width=3 if won else 1,
                )
        if len(winning_line) >= 2:
            (first_x, first_y), (last_x, last_y) = winning_line[0], winning_line[-1]
            self.canvas.create_line(
                margin + first_y * size, margin + first_x * size, margin + last_y * size, margin + last_x * size,
                fill="#e11d48", width=4, capstyle=tk.ROUND,
            )
        if self.game and self.current_ply:
            x, y = _log_move_coordinates(self.game.moves[self.current_ply - 1])
            cx, cy = margin + y * size, margin + x * size
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#22c55e", outline="white")

    def _update_result_label(self) -> None:
        if self.game is None:
            self.result_label.configure(text="")
            return
        data = self.game.data
        winner = data.get("winner")
        players = data.get("players", {})
        if winner:
            color = "Black" if winner == players.get("black") else "White" if winner == players.get("white") else ""
            result = f"Winner: {winner}" + (f" ({color})" if color else "")
        else:
            result = "Result: Draw" if data.get("result") == "draw" else "No winner information"
        reason = data.get("reason")
        if reason:
            result += f"  |  {reason}"
        if self.current_ply < len(self.game.moves):
            result = f"Replaying move {self.current_ply}/{len(self.game.moves)}  |  " + result
        self.result_label.configure(text=result)

    def _on_move_selected(self, _event: tk.Event) -> None:
        selection = self.move_list.curselection()
        if selection:
            self.current_ply = selection[0] + 1
            self._render_current_position()

    def _previous_move(self) -> None:
        if self.game and self.current_ply > 0:
            self.current_ply -= 1
            self._render_current_position()

    def _next_move(self) -> None:
        if self.game and self.current_ply < len(self.game.moves):
            self.current_ply += 1
            self._render_current_position()
        elif self.game and self.game.moves:
            self.current_ply = 1
            self._render_current_position()

    def _previous_game(self) -> None:
        self._refresh_game_files()
        if self.game_files:
            self._load_game_at(self.game_index + 1)

    def _next_game(self) -> None:
        self._refresh_game_files()
        if self.game_files:
            self._load_game_at(self.game_index - 1)

    def _on_enter(self, _event: tk.Event) -> str:
        self._next_game()
        return "break"

    def _on_space(self, _event: tk.Event) -> str:
        self._next_move()
        return "break"

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


# =====================================================================
# GUI: APP CLIENT
# =====================================================================
class ClientApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gomoku - Client")
        self.root.minsize(1000, 720)
        self.root.configure(bg=PAGE_BG)

        self.events: queue.Queue[tuple] = queue.Queue()
        self.session = 0  # tăng mỗi lần rời phòng; sự kiện của phiên cũ bị bỏ qua
        self.client: GomokuClient | None = None
        self.state: dict[str, Any] | None = None
        self.state_at = 0.0
        self.connected = True
        self.bot: WrapperBot | None = None
        self.loaded_bot: WrapperBot | None = None  # bot đã tải xong; None thì chưa được mở hoặc vào phòng
        self.loaded_bot_source: tuple[str, float] | None = None  # (đường dẫn, mtime) của file đã tải
        self.load_generation = 0  # tăng mỗi lần tải; kết quả của lần tải cũ bị bỏ qua
        self.auto_name = ""  # tên đang được điền tự động; người dùng tự nhập tên khác thì không ghi đè
        self.bot_key: tuple[int, int] | None = None
        self.logged_moves = (0, 0)  # (game_no, số nước đã đưa vào move log)
        self.drawn_key: tuple | None = None
        self.saved_game = 0
        self.saved_name: str | None = None
        self.log_dir = DEFAULT_LOG_DIR
        self.viewer_window: tk.Toplevel | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._draw_board()
        self._show_start()
        self.root.after(100, self._tick)

    # ------------------------------ giao diện ------------------------------
    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=NAVY, padx=16, pady=10)
        header.pack(fill="x")
        self.title_label = tk.Label(header, text="GOMOKU · CLIENT", bg=NAVY, fg="white", font=("Segoe UI", 17, "bold"))
        self.title_label.pack(side="left")
        self.turn_label = tk.Label(header, text="", bg=NAVY, fg="#dce8f2", font=("Segoe UI", 10, "bold"))
        self.turn_label.pack(side="right")

        self.status = tk.Label(self.root, text="", anchor="w", bg="#dbe4ec", fg="#334155", padx=10, pady=6, font=("Segoe UI", 9))
        self.status.pack(fill="x", side="bottom")

        self._build_start_screen()
        self.main_view = tk.Frame(self.root, bg=PAGE_BG, padx=14, pady=8)
        board_panel = tk.Frame(self.main_view, bg=PAGE_BG)
        board_panel.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(board_panel, width=BOARD_EXTENT, height=BOARD_EXTENT, bg=BOARD_BG, highlightthickness=1, highlightbackground="#9d7845")
        self.canvas.pack(anchor="center", pady=8)
        self.result_label = tk.Label(board_panel, text="", bg=PAGE_BG, fg=NAVY, font=("Segoe UI", 12, "bold"), wraplength=BOARD_EXTENT)
        self.result_label.pack(pady=(4, 0))
        self.side_panel = tk.Frame(self.main_view, bg=PANEL_BG, padx=12, pady=12, width=360)
        self.side_panel.pack(side="right", fill="y", padx=(14, 0))
        self.side_panel.pack_propagate(False)
        self._build_setup()
        self._build_room()

    def _build_start_screen(self) -> None:
        self.start_screen = tk.Frame(self.root, bg=PAGE_BG)
        center = tk.Frame(self.start_screen, bg=PAGE_BG)
        center.place(relx=0.5, rely=0.45, anchor="center")
        tk.Label(center, text="Bạn muốn làm gì?", bg=PAGE_BG, fg=NAVY, font=("Segoe UI", 16, "bold")).pack(pady=(0, 16))
        choices = (
            ("Client", "Vào phòng của server bằng địa chỉ host và match code.", self._show_setup),
            ("Xem log", "Phát lại các ván đã lưu trong folder log.", self._open_logs),
        )
        for title, description, command in choices:
            card = tk.Frame(center, bg=PANEL_BG, padx=16, pady=12, highlightthickness=1, highlightbackground="#cbd5e1")
            card.pack(fill="x", pady=6)
            tk.Button(
                card, text=title, command=command, width=12, bg=NAVY, fg="white", activebackground="#2c5278",
                activeforeground="white", relief="flat", cursor="hand2", font=("Segoe UI", 12, "bold"), pady=6,
            ).pack(side="left")
            tk.Label(card, text=description, bg=PANEL_BG, fg="#334155", font=("Segoe UI", 10), wraplength=330, justify="left", anchor="w", width=48).pack(side="left", padx=(14, 0))

        folder_row = tk.Frame(center, bg=PAGE_BG)
        folder_row.pack(fill="x", pady=(14, 0))
        tk.Label(folder_row, text="Folder log:", bg=PAGE_BG, fg=NAVY).pack(side="left")
        ttk.Button(folder_row, text="Chọn...", command=self._choose_folder).pack(side="right")
        self.folder_entry = ttk.Entry(folder_row)
        self.folder_entry.insert(0, str(DEFAULT_LOG_DIR))
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=8)

    def _form_row(self, parent: tk.Widget, row: int, label: str, widget: tk.Widget) -> tk.Label:
        caption = tk.Label(parent, text=label, bg=PANEL_BG, fg=NAVY, anchor="w")
        caption.grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        return caption

    def _hint(self, parent: tk.Widget, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=PANEL_BG, fg="#64748b", font=("Segoe UI", 9), wraplength=320, justify="left", anchor="w")

    def _build_bot_form(self) -> None:
        """Phần chọn bot, giống hệt nhau ở server và client: chọn file là tải ngay."""
        self.setup = tk.Frame(self.side_panel, bg=PANEL_BG)
        tk.Label(self.setup, text="BOT", bg=PANEL_BG, fg=NAVY, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        form = tk.Frame(self.setup, bg=PANEL_BG)
        form.pack(fill="x", pady=(4, 10))
        form.columnconfigure(1, weight=1)
        self.name_entry = ttk.Entry(form)
        self._form_row(form, 0, "Tên bot:", self.name_entry)
        self.wrapper_row = tk.Frame(form, bg=PANEL_BG)
        self.wrapper_entry = ttk.Entry(self.wrapper_row)
        self.wrapper_entry.insert(0, str(DEFAULT_WRAPPER))
        ttk.Button(self.wrapper_row, text="...", width=3, command=self._choose_wrapper).pack(side="right", padx=(4, 0))
        self.wrapper_entry.pack(side="left", fill="x", expand=True)
        self.wrapper_entry.bind("<KeyRelease>", self._on_wrapper_edited)
        self.wrapper_entry.bind("<Return>", lambda _event: self._load_bot())
        self.wrapper_entry.bind("<FocusOut>", self._on_wrapper_focus_out)
        self._form_row(form, 1, "File bot:", self.wrapper_row)
        self.bot_load_row = tk.Frame(form, bg=PANEL_BG)
        self.bot_load_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(self.bot_load_row, text="Tải lại", width=7, command=lambda: self._load_bot(force=True)).pack(side="right", anchor="n", padx=(4, 0))
        self.bot_load_label = tk.Label(self.bot_load_row, text="", bg=PANEL_BG, font=("Segoe UI", 9), wraplength=250, justify="left", anchor="w")
        self.bot_load_label.pack(side="left", fill="x", expand=True)
        self._apply_default_name()

    def _build_setup(self) -> None:
        self._build_bot_form()

        self.client_form = tk.Frame(self.setup, bg=PANEL_BG)
        self.client_form.columnconfigure(1, weight=1)
        self.client_form.pack(fill="x")
        tk.Label(self.client_form, text="PHÒNG CẦN VÀO", bg=PANEL_BG, fg=NAVY, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.host_entry = ttk.Entry(self.client_form)
        self._form_row(self.client_form, 1, "Host:", self.host_entry)
        self._hint(self.client_form, "Ví dụ http://100.x.y.z:8765 (Tailscale) hoặc https://abc.ngrok-free.app").grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.join_code_entry = ttk.Entry(self.client_form)
        self._form_row(self.client_form, 3, "Match code:", self.join_code_entry)
        self._hint(self.client_form, "Bot phải tải xong thì mới vào phòng được. Vào phòng là ván tự bắt đầu, luật do server đặt.").grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 6))

        setup_buttons = tk.Frame(self.setup, bg=PANEL_BG)
        setup_buttons.pack(fill="x", side="bottom")
        ttk.Button(setup_buttons, text="← Quay lại", command=self._show_start).pack(side="left")
        self.go_button = ttk.Button(setup_buttons, text="Vào phòng", command=self._join_match)
        self.go_button.pack(side="right", fill="x", expand=True, padx=(8, 0))

    def _build_room(self) -> None:
        self.room = tk.Frame(self.side_panel, bg=PANEL_BG)
        self.room_status = tk.Label(self.room, text="", bg=PANEL_BG, font=("Segoe UI", 11, "bold"), anchor="w")
        self.room_status.pack(fill="x")
        self.rules_label = tk.Label(self.room, text="", bg=PANEL_BG, fg="#475569", font=("Segoe UI", 9), anchor="w")
        self.rules_label.pack(fill="x", pady=(0, 6))

        self.cards = {stone: self._build_card(stone) for stone in (BLACK, WHITE)}

        buttons = tk.Frame(self.room, bg=PANEL_BG)
        buttons.pack(fill="x", pady=(2, 8))
        ttk.Button(buttons, text="Xem log", command=self._open_logs).pack(side="left")
        self.leave_button = ttk.Button(buttons, text="Rời phòng", command=self._leave)
        self.leave_button.pack(side="right")

        tk.Label(self.room, text="MOVE LOG", bg=PANEL_BG, fg=NAVY, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.move_list = tk.Listbox(self.room, font=("Consolas", 10), activestyle="none", bg="white", fg="#1e293b", relief="solid", borderwidth=1)
        self.move_list.pack(fill="both", expand=True, pady=(4, 0))

    def _build_card(self, stone: int) -> dict[str, tk.Widget]:
        card = tk.Frame(self.room, bg=PANEL_BG, padx=8, pady=5, highlightthickness=1, highlightbackground="#cbd5e1")
        card.pack(fill="x", pady=(0, 6))
        top = tk.Frame(card, bg=PANEL_BG)
        top.pack(fill="x")
        dot = tk.Canvas(top, width=18, height=18, bg=PANEL_BG, highlightthickness=0)
        dot.create_oval(2, 2, 16, 16, fill=STONE_COLORS[stone], outline=STONE_OUTLINES[stone])
        dot.pack(side="left")
        total = tk.Label(top, text="", bg=PANEL_BG, fg="#1e293b", font=("Consolas", 18, "bold"))
        total.pack(side="right")  # đóng gói trước tên: tên dài bị cắt bớt thay vì đẩy đồng hồ ra ngoài
        name = tk.Label(top, text="", bg=PANEL_BG, fg="#1e293b", font=("Segoe UI", 10, "bold"), anchor="w")
        name.pack(side="left", padx=(4, 0))
        bottom = tk.Frame(card, bg=PANEL_BG)
        bottom.pack(fill="x")
        online = tk.Label(bottom, text="", bg=PANEL_BG, fg="#64748b", font=("Segoe UI", 9), anchor="w")
        online.pack(side="left")
        per_move = tk.Label(bottom, text="", bg=PANEL_BG, fg="#475569", font=("Consolas", 10))
        per_move.pack(side="right")
        return {"card": card, "top": top, "bottom": bottom, "dot": dot, "name": name, "total": total, "online": online, "per_move": per_move}

    # ------------------------------ chuyển màn hình ------------------------------
    def _show_start(self) -> None:
        self.main_view.pack_forget()
        self.turn_label.configure(text="")
        self.start_screen.pack(fill="both", expand=True)

    def _show_setup(self) -> None:
        self.log_dir = self._selected_log_dir()
        self.start_screen.pack_forget()
        self.room.pack_forget()
        self._load_bot()
        self._update_go_button()
        self.setup.pack(fill="both", expand=True)
        self.main_view.pack(fill="both", expand=True)
        self._set_status("")

    def _selected_log_dir(self) -> Path:
        return Path(self.folder_entry.get().strip() or DEFAULT_LOG_DIR)

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self.root, title="Chọn folder log", initialdir=str(self._selected_log_dir()))
        if folder:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)

    def _open_logs(self) -> None:
        if self.viewer_window is not None and self.viewer_window.winfo_exists():
            self.viewer_window.destroy()  # mở lại để thấy các log mới
        self.viewer_window = tk.Toplevel(self.root)
        LogViewer(self.viewer_window, self.log_dir if self.state is not None else self._selected_log_dir())

    def _apply_default_name(self) -> None:
        """Điền tên theo BOT_NAME trong file bot, trừ khi người dùng đã tự nhập tên."""
        default = read_bot_name(Path(self.wrapper_entry.get().strip())) or "Bot"
        if self.name_entry.get().strip() in ("", self.auto_name):
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, default)
        self.auto_name = default

    def _choose_wrapper(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Chọn file wrapper của bot",
            initialdir=str(BASE_DIR),
            filetypes=(("Python files", "*.py"), ("All files", "*.*")),
        )
        if filename:
            self.wrapper_entry.delete(0, tk.END)
            self.wrapper_entry.insert(0, filename)
            self._apply_default_name()
            self._load_bot()

    # ------------------------------ tải bot, vào phòng ------------------------------
    def _setup_failed(self, message: str) -> None:
        self._update_go_button()
        self._set_status(message)

    def _update_go_button(self) -> None:
        """Chỉ được bấm Mở phòng / Vào phòng khi bot đã sẵn sàng."""
        self.go_button.state(["!disabled"] if self.loaded_bot is not None else ["disabled"])

    def _show_bot_load(self, text: str, color: str) -> None:
        self.bot_load_label.configure(text=text, fg=color)

    def _on_wrapper_edited(self, _event: tk.Event) -> None:
        """Đang sửa đường dẫn: bot cũ không còn đúng với file trong ô, nên khoá nút cho tới khi tải file mới."""
        if self.loaded_bot_source is None or self.wrapper_entry.get().strip() != self.loaded_bot_source[0]:
            self.load_generation += 1
            self.loaded_bot, self.loaded_bot_source = None, None
            self._show_bot_load("○ Nhấn Enter để tải bot.", "#b45309")
            self._update_go_button()

    def _on_wrapper_focus_out(self, _event: tk.Event) -> None:
        if self.loaded_bot_source is None and self.bot_load_label.cget("text").startswith("○ Nhấn"):
            self._load_bot()

    def _load_bot(self, force: bool = False) -> None:
        """Tải file wrapper ngay khi chọn. Nút Mở phòng / Vào phòng chỉ mở khi bot báo sẵn sàng."""
        path = Path(self.wrapper_entry.get().strip())
        self._apply_default_name()
        try:
            source = (str(self.wrapper_entry.get().strip()), path.stat().st_mtime)
        except OSError:
            source = None
        if not force and source is not None and source == self.loaded_bot_source and self.loaded_bot is not None:
            self._update_go_button()  # file không đổi: giữ bot đã tải, khỏi tải lại model
            return
        self.load_generation += 1
        generation = self.load_generation
        self.loaded_bot, self.loaded_bot_source = None, None
        self._update_go_button()
        if source is None or not path.is_file():
            self._show_bot_load(f"✕ Không tìm thấy file bot: {path}", "#b91c1c")
            return
        self._show_bot_load(f"○ Đang tải {path.name}...", "#b45309")

        def post(action: Callable[[], None]) -> None:
            def guarded() -> None:
                if generation == self.load_generation:
                    action()

            self.events.put(("call", guarded))

        def work() -> None:
            try:
                bot = WrapperBot(path)

                def on_ready() -> None:
                    def apply() -> None:
                        self.loaded_bot, self.loaded_bot_source = bot, source
                        self._show_bot_load("● Bot đã sẵn sàng.", "#15803d")
                        self._update_go_button()

                    post(apply)

                room = BotRoom(lambda text: post(lambda: self._show_bot_load(f"○ {text}", "#b45309")), on_ready)
                bot.prepare(room)
                room.ready()  # load_bot không gọi room.ready() thì coi như sẵn sàng khi hàm kết thúc
            except Exception as error:  # noqa: BLE001  (lỗi của file bot phải hiện lên GUI)
                traceback.print_exc()
                message = f"✕ Lỗi khi tải {path.name}: {error}"

                def failed() -> None:
                    self.loaded_bot, self.loaded_bot_source = None, None
                    self._show_bot_load(message, "#b91c1c")
                    self._update_go_button()

                post(failed)

        threading.Thread(target=work, daemon=True).start()

    def _join_match(self) -> None:
        code, name = self.join_code_entry.get(), _clean_name(self.name_entry.get(), "Bot")
        try:
            client = GomokuClient(self.host_entry.get())
        except ValueError as error:
            self._set_status(str(error))
            return

        bot = self.loaded_bot
        if bot is None:  # phòng hờ: nút đã bị khoá khi bot chưa sẵn sàng
            self._set_status("Bot chưa sẵn sàng, chưa thể vào phòng.")
            return
        self.go_button.state(["disabled"])
        self._set_status(f"Đang kết nối {client.url}...")

        def work() -> None:
            try:
                client.join(code, name)
                send_ready(client)
            except (ApiError, ConnectionLost) as error:
                message = f"Không vào được phòng: {error}"
                self.events.put(("call", lambda: self._setup_failed(message)))
                return
            self.events.put(("call", lambda: self._joined(client, bot)))

        threading.Thread(target=work, daemon=True).start()

    def _joined(self, client: GomokuClient, bot: WrapperBot) -> None:
        self._update_go_button()
        self.bot = bot
        self._enter_room(client)
        self._set_status(f"Đã vào phòng tại {client.url}. Ván sẽ tự bắt đầu.")

    def _enter_room(self, client: GomokuClient) -> None:
        self.session += 1
        self.client = client
        self.state = None
        self.connected = True
        self.bot_key = None
        self.logged_moves = (0, 0)
        self.drawn_key = None
        self.saved_game = 0
        self.saved_name = None
        self.move_list.delete(0, tk.END)
        self.result_label.configure(text="")
        self.setup.pack_forget()
        self.room.pack(fill="both", expand=True)
        threading.Thread(target=self._poll_loop, args=(self.session, client), daemon=True).start()

    # ------------------------------ mạng ------------------------------
    def _poll_loop(self, session: int, client: GomokuClient) -> None:
        since = -1
        while session == self.session:
            try:
                state = client.state(since, LONG_POLL_WAIT)
            except ConnectionLost as error:
                self.events.put(("lost", session, str(error)))
                time.sleep(1.5)
                continue
            except ApiError as error:
                self.events.put(("ended", session, error.message))
                return
            since = state["version"]
            self.events.put(("state", session, state))
            if state["status"] == CLOSED:
                return

    def _tick(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        self._update_clocks()
        self.root.after(100, self._tick)

    def _handle_event(self, event: tuple) -> None:
        kind = event[0]
        if kind == "call":
            event[1]()
            return
        if event[1] != self.session:
            return
        if kind == "state":
            reconnected = not self.connected
            self.connected = True
            self._apply_state(event[2])
            if reconnected and self.state is not None:
                self._set_status("Đã kết nối lại. " + self._status_message(self.state))
        elif kind == "lost":
            if self.connected:
                self.connected = False
                self._set_status(f"Mất kết nối, đang thử lại: {event[2]}")
            self._update_turn_label()
        elif kind == "ended":
            self._set_status(f"Phiên đã kết thúc: {event[2]}")
            self.result_label.configure(text=event[2])
        elif kind == "error":
            self._set_status(event[2])
        elif kind == "bot_retry":
            self.bot_key = None
            self._set_status(event[2])

    # ------------------------------ đồng bộ trạng thái ------------------------------
    def _apply_state(self, state: dict[str, Any]) -> None:
        previous = self.state
        if previous is not None and state["version"] < previous["version"]:
            return  # câu trả lời đến trễ, cũ hơn cái đang có
        self.state, self.state_at = state, time.perf_counter()
        if state["result"] and self.saved_game != state["game_no"]:
            self.saved_game = state["game_no"]
            self.saved_name = save_game_log(state, self.log_dir)  # client giữ một bản log riêng
        if previous is None or previous["status"] != state["status"]:
            self._set_status(self._status_message(state))
        self._sync_move_log(state)
        self._refresh_room(state)
        key = (state["game_no"], state["ply"], state["status"])
        if key != self.drawn_key:
            self.drawn_key = key
            self._draw_board()
        self._update_turn_label()
        self._update_clocks()
        if self.bot is not None:
            self._drive_bot(state)

    def _status_message(self, state: dict[str, Any]) -> str:
        status, series = state["status"], state["series"]
        if status == LOADING:
            return "Đang báo sẵn sàng..."
        if status == READY:
            return "Đã đủ 2 bot. Ván sẽ tự bắt đầu."
        if status == PLAYING:
            return f"Ván {series['game']}/{series['games']} đang diễn ra."
        if status == CLOSED:
            return "Server đã đóng phòng."
        result = state["result"] or {}
        text = result.get("text", "Ván đã kết thúc.")
        return f"{text} Log: {self.saved_name}" if self.saved_name else text

    def _sync_move_log(self, state: dict[str, Any]) -> None:
        game_no, count = self.logged_moves
        if game_no != state["game_no"]:
            self.move_list.delete(0, tk.END)
            count = 0
        for move in state["moves"][count:]:
            color = "Đen" if move["player"] == BLACK else "Trắng"
            name = state["players"][str(move["player"])]["name"] or ""
            self.move_list.insert(tk.END, f"{move['ply']:03d} {color:<5} {name[:12]:<12} ({move['x']:2d},{move['y']:2d}) {move['spent']:5.1f}s")
        self.logged_moves = (state["game_no"], len(state["moves"]))
        if state["moves"]:
            self.move_list.see(tk.END)

    def _refresh_room(self, state: dict[str, Any]) -> None:
        status, series = state["status"], state["series"]
        self.room_status.configure(text=STATUS_TEXT[status], fg=STATUS_COLOR[status])
        total, per_move = state["rules"]["total_time"], state["rules"]["move_time"]
        total_text = f"{total / 60:g} phút/bên" if total else "không giới hạn"
        move_text = f"{per_move:g} s" if per_move else "không giới hạn"
        self.rules_label.configure(text=f"Lượt {series['round']} · Ván {series['game']}/{series['games']} · {total_text} · Mỗi nước: {move_text}")
        for stone, card in self.cards.items():
            player = state["players"][str(stone)]
            you = " (bạn)" if stone == state["you"] else ""
            card["name"].configure(text=(player["name"] or "Đang chờ...") + you)
            loading = "chưa sẵn sàng" if player["name"] and not player["ready"] else ""
            card["online"].configure(
                text=("● online" if player["online"] else ("○ offline" if player["name"] else ""))
                + (f" · thắng {player['wins']}" if player["name"] else "")
                + (f" · {loading}" if loading else ""),
                fg="#15803d" if player["online"] else "#b91c1c",
            )
            background = ACTIVE_BG if state["to_move"] == stone else PANEL_BG
            for widget_name in ("card", "top", "bottom", "dot", "name", "total", "online", "per_move"):
                card[widget_name].configure(bg=background)

        result = state["result"]
        result_text = result["text"] if result else ""
        if result and series["over"]:  # chỉ một dòng để cửa sổ không cao thêm; chi tiết ván nằm ở thanh trạng thái
            black, white = state["players"][str(BLACK)], state["players"][str(WHITE)]
            black_name, white_name = (black["name"] or "Đen")[:16], (white["name"] or "Trắng")[:16]
            result_text = f"Hết lượt: {black_name} {black['wins']} - {white['wins']} {white_name}, hoà {series['draws']}."
        self.result_label.configure(text=result_text)

    def _update_clocks(self) -> None:
        state = self.state
        if state is None:
            return
        since_update = time.perf_counter() - self.state_at
        running = state["status"] == PLAYING
        elapsed = since_update if running else 0.0
        next_in = state["series"]["next_game_in"]
        if next_in is not None:
            left_to_next = next_in - since_update
            label = "VÁN ĐẦU BẮT ĐẦU" if state["status"] == READY else "VÁN TIẾP THEO"
            self.room_status.configure(text=f"{label} SAU {left_to_next:.0f} S" if left_to_next > 0 else "CHỜ CẢ HAI BÊN ONLINE")
        for stone, card in self.cards.items():
            left = state["players"][str(stone)]["total_left"]
            active = state["to_move"] == stone
            if left is not None and active:
                left = max(0.0, left - elapsed)
            per_move_text = ""
            move_low = False
            if active and running and state["move_left"] is not None:
                move_left = max(0.0, state["move_left"] - elapsed)
                per_move_text = f"nước này {fmt_clock(move_left)}"
                move_low = move_left < LOW_TIME_SECONDS
            total_low = left is not None and left < LOW_TIME_SECONDS
            card["total"].configure(text=fmt_clock(left), fg="#dc2626" if total_low else "#1e293b")
            card["per_move"].configure(text=per_move_text, fg="#dc2626" if move_low else "#475569")

    def _update_turn_label(self) -> None:
        state = self.state
        if state is None:
            self.turn_label.configure(text="")
        elif not self.connected:
            self.turn_label.configure(text="MẤT KẾT NỐI")
        elif state["status"] == PLAYING:
            name = state["players"][str(state["to_move"])]["name"]
            self.turn_label.configure(text="Lượt của bot của bạn" if state["your_turn"] else f"Lượt của {name}")
        else:
            self.turn_label.configure(text=STATUS_TEXT[state["status"]])

    # ------------------------------ bàn cờ ------------------------------
    def _draw_board(self) -> None:
        self.canvas.delete("all")
        end = MARGIN + CELL * (BOARD_SIZE - 1)
        for index in range(BOARD_SIZE):
            pos = MARGIN + index * CELL
            self.canvas.create_line(MARGIN, pos, end, pos, fill="#624b2c")
            self.canvas.create_line(pos, MARGIN, pos, end, fill="#624b2c")
            self.canvas.create_text(pos, MARGIN - 23, text=str(index), fill="#4b3b26", font=("Segoe UI", 8))
            self.canvas.create_text(MARGIN - 23, pos, text=str(index), fill="#4b3b26", font=("Segoe UI", 8))
        state = self.state
        if state is None:
            return
        if state["status"] in (READY, PLAYING) and state["ply"] == 0:
            # Vùng giữa không được đi nước 1: viền đứt nét quanh vùng cấm.
            low = MARGIN + (OUTER_RING_WIDTH - 0.5) * CELL
            high = MARGIN + (BOARD_SIZE - OUTER_RING_WIDTH - 0.5) * CELL
            self.canvas.create_rectangle(low, low, high, high, outline="#b91c1c", dash=(6, 4), width=2)
        result = state["result"]
        winning_line = [tuple(cell) for cell in (result["line"] if result else [])]
        winning_cells = set(winning_line)
        radius = CELL * 0.38
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                stone = state["board"][x][y]
                if not stone:
                    continue
                cx, cy = MARGIN + y * CELL, MARGIN + x * CELL
                won = (x, y) in winning_cells
                self.canvas.create_oval(
                    cx - radius, cy - radius, cx + radius, cy + radius,
                    fill=STONE_COLORS[stone],
                    outline="#e11d48" if won else STONE_OUTLINES[stone],
                    width=3 if won else 1,
                )
        if len(winning_line) >= 2:
            (first_x, first_y), (last_x, last_y) = winning_line[0], winning_line[-1]
            self.canvas.create_line(
                MARGIN + first_y * CELL, MARGIN + first_x * CELL, MARGIN + last_y * CELL, MARGIN + last_x * CELL,
                fill="#e11d48", width=4, capstyle=tk.ROUND,
            )
        last = state["last_move"]
        if last:
            cx, cy = MARGIN + last["y"] * CELL, MARGIN + last["x"] * CELL
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#22c55e", outline="white")

    # ------------------------------ bot ------------------------------
    def _drive_bot(self, state: dict[str, Any]) -> None:
        bot, client, session = self.bot, self.client, self.session
        key = (state["game_no"], state["ply"])
        if not state["your_turn"] or key == self.bot_key:
            return
        self.bot_key = key
        received_at = self.state_at
        self._set_status("Bot của bạn đang suy nghĩ...")

        def work() -> None:
            try:
                x, y = bot.choose(state, time.perf_counter() - received_at)
            except Exception as error:  # noqa: BLE001  (lỗi của file bot phải hiện lên GUI)
                traceback.print_exc()
                self.events.put(("bot_retry", session, f"Bot lỗi: {error}"))
                return
            for _ in range(5):
                try:
                    reply = client.move(x, y, state["ply"], state["game_no"])
                except ConnectionLost as error:
                    self.events.put(("error", session, f"Gửi nước đi thất bại, thử lại: {error}"))
                    time.sleep(0.5)
                    continue
                except ApiError as error:
                    self.events.put(("bot_retry", session, f"Nước ({x}, {y}) bị từ chối: {error.message}"))
                    return
                self.events.put(("call", lambda: self._set_status(f"Bot của bạn đi ({x}, {y}).")))
                self.events.put(("state", session, reply))
                return
            self.events.put(("bot_retry", session, "Không gửi được nước đi, sẽ thử lại khi có kết nối."))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------ rời phòng ------------------------------
    def _confirm_leave_if_playing(self) -> bool:
        if self.state is None or self.state["status"] != PLAYING:
            return True
        return messagebox.askyesno("Rời phòng", "Ván đang chơi. Rời phòng sẽ bị tính thua. Tiếp tục?", parent=self.root)

    def _leave(self) -> None:
        if not self._confirm_leave_if_playing():
            return
        self._disconnect()
        self.state = None
        self.drawn_key = None
        self.result_label.configure(text="")
        self._draw_board()
        self._set_status("Đã rời phòng.")
        self._show_start()

    def _disconnect(self) -> threading.Thread | None:
        """Rời phòng ở luồng nền, vì mạng có thể đang chậm."""
        self.session += 1
        client = self.client
        self.client = None

        def work() -> None:
            if client is not None:
                try:
                    client.leave()
                except (ApiError, ConnectionLost):
                    pass

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        return thread

    def _on_close(self) -> None:
        if not self._confirm_leave_if_playing():
            return
        thread = self._disconnect()
        if thread is not None:
            thread.join(timeout=2.0)
        self.root.destroy()

    def _set_status(self, message: str) -> None:
        self.status.configure(text=message)


# =====================================================================
# CHẠY CHƯƠNG TRÌNH
# =====================================================================
def main() -> None:
    if tk is None:
        sys.exit("Máy này không có tkinter. Hãy chạy bot không giao diện bằng online_wrapper_template.py.")
    root = tk.Tk()
    ClientApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
