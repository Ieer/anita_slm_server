"""anita_slm_server 套件入口。

原 `agent_slm_server` 套件已更名為 `anita_slm_server`。
為了向後相容，若外部仍嘗試 import 舊名稱，建議在升級時同步調整引用。
"""

from .qwenchat import QwenChatAPI  # noqa: F401
