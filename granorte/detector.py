"""
╔══════════════════════════════════════════════════════════════════════════╗
║         SISTEMA DE MONITORAMENTO DE CAMINHÕES — ANÁLISE DE VÍDEO        ║
║         Vídeo 1: Placa  |  Vídeo 2: Basculante (limpo/sujo)            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  INSTALAÇÃO:                                                             ║
║    pip install ultralytics easyocr opencv-python numpy                   ║
║                                                                          ║
║  USO:                                                                    ║
║    # Informar os dois vídeos diretamente                                 ║
║    python detector.py --video-placa placa.mp4 \                         ║
║                       --video-basculante basculante.mp4                 ║
║                                                                          ║
║    # Sem argumentos: o sistema pede os caminhos no terminal             ║
║    python detector.py                                                    ║
║                                                                          ║
║    # Salvar vídeo de saída com as detecções                             ║
║    python detector.py --video-placa p.mp4 --video-basculante b.mp4 \   ║
║                       --salvar-saida                                    ║
║                                                                          ║
║  CONTROLES NA JANELA:                                                    ║
║    ESPAÇO     = pausar / retomar                                         ║
║    → (seta)   = avançar 5 segundos                                      ║
║    ← (seta)   = voltar 5 segundos                                       ║
║    + / -      = velocidade de reprodução                                 ║
║    S          = salvar frame atual como imagem                           ║
║    Q / ESC    = sair                                                     ║
║                                                                          ║
║  SAÍDA:                                                                  ║
║    • registros/historico.db  (SQLite com todo o histórico)              ║
║    • registros/placas/       (frames das placas detectadas)             ║
║    • registros/basculantes/  (frames do basculante)                     ║
║    • saida_monitoramento.mp4 (vídeo final, com --salvar-saida)          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import re
import sqlite3
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ── Tentar carregar YOLO (opcional — usado só para câmera de placa) ─────────
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

# ── Tentar carregar EasyOCR ──────────────────────────────────────────────────
try:
    import easyocr
    EASYOCR_OK = True
except ImportError:
    EASYOCR_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# BANCO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════
class Banco:
    def __init__(self, pasta: str):
        Path(pasta).mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(Path(pasta) / "historico.db"),
            check_same_thread=False,
        )
        self._criar_tabelas()

    def _criar_tabelas(self):
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS placas (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                placa     TEXT NOT NULL,
                tipo      TEXT,
                confianca REAL,
                foto      TEXT
            );
            CREATE TABLE IF NOT EXISTS basculantes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status    TEXT NOT NULL,
                score     REAL,
                placa_ref TEXT,
                foto      TEXT
            );
            CREATE TABLE IF NOT EXISTS entradas (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                placa           TEXT NOT NULL,
                status_basculante TEXT,
                observacao      TEXT
            );
        """)
        self.conn.commit()

    def salvar_placa(self, placa, tipo, confianca, foto=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            "INSERT INTO placas(timestamp,placa,tipo,confianca,foto) VALUES(?,?,?,?,?)",
            (ts, placa, tipo, confianca, foto),
        )
        self.conn.commit()
        return ts

    def salvar_basculante(self, status, score, placa_ref="", foto=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            "INSERT INTO basculantes(timestamp,status,score,placa_ref,foto) VALUES(?,?,?,?,?)",
            (ts, status, score, placa_ref, foto),
        )
        self.conn.commit()
        return ts

    def registrar_entrada(self, placa, status_basculante, obs=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            "INSERT INTO entradas(timestamp,placa,status_basculante,observacao) VALUES(?,?,?,?)",
            (ts, placa, status_basculante, obs),
        )
        self.conn.commit()

    def historico_placas(self, limite=20):
        cur = self.conn.execute(
            "SELECT timestamp,placa,tipo,confianca FROM placas ORDER BY id DESC LIMIT ?",
            (limite,),
        )
        return cur.fetchall()

    def historico_basculantes(self, limite=20):
        cur = self.conn.execute(
            "SELECT timestamp,status,score,placa_ref FROM basculantes ORDER BY id DESC LIMIT ?",
            (limite,),
        )
        return cur.fetchall()

    def historico_entradas(self, limite=10):
        cur = self.conn.execute(
            "SELECT timestamp,placa,status_basculante FROM entradas ORDER BY id DESC LIMIT ?",
            (limite,),
        )
        return cur.fetchall()


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR DE PLACA
# ═══════════════════════════════════════════════════════════════════════════════
PADRAO_MERCOSUL = re.compile(r"^[A-Z]{3}\d[A-Z]\d{2}$")
PADRAO_ANTIGA   = re.compile(r"^[A-Z]{3}\d{4}$")


def validar_placa(txt: str):
    t = re.sub(r"[^A-Z0-9]", "", txt.upper())
    if PADRAO_MERCOSUL.match(t):
        return t, "Mercosul"
    if PADRAO_ANTIGA.match(t):
        return t, "Antiga"
    return None, None


def preprocessar_para_ocr(recorte: np.ndarray) -> np.ndarray:
    h_alvo = 80
    r = h_alvo / max(recorte.shape[0], 1)
    recorte = cv2.resize(recorte, (int(recorte.shape[1] * r), h_alvo))
    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    cinza = clahe.apply(cinza)
    _, thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


class DetectorPlaca:
    def __init__(self, modelo_path: str | None = None):
        self.yolo = None
        if modelo_path and YOLO_OK:
            p = Path(modelo_path)
            if p.exists():
                print(f"[INFO] Carregando YOLO placa: {modelo_path}")
                self.yolo = YOLO(str(p))

        self.ocr = None
        if EASYOCR_OK:
            print("[INFO] Carregando EasyOCR...")
            self.ocr = easyocr.Reader(["pt", "en"], gpu=False, verbose=False)
            print("[INFO] EasyOCR pronto.")
        else:
            print("[AVISO] EasyOCR não instalado — OCR desativado.")

        self.historico: deque[str] = deque(maxlen=12)
        self._ultimo_ocr: float = 0.0

    def detectar(self, frame: np.ndarray, intervalo_ocr: float = 1.5):
        """Retorna lista de (x1,y1,x2,y2, placa_texto, tipo, confianca)."""
        regioes = []

        if self.yolo:
            resultados = self.yolo.track(frame, conf=0.40, persist=True, verbose=False)
            for res in resultados:
                for box in (res.boxes or []):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    regioes.append((x1, y1, x2, y2, conf))
        else:
            # Fallback: busca por regiões retangulares claras (placa)
            regioes = self._detectar_sem_modelo(frame)

        resultado = []
        agora = time.perf_counter()
        fazer_ocr = (agora - self._ultimo_ocr) > intervalo_ocr

        for item in regioes:
            x1, y1, x2, y2, conf = item
            placa_txt, tipo = None, None

            if fazer_ocr and self.ocr:
                recorte = frame[max(0, y1):y2, max(0, x1):x2]
                if recorte.size > 0:
                    proc = preprocessar_para_ocr(recorte)
                    leituras = self.ocr.readtext(
                        proc, detail=1,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
                    )
                    txt = "".join(r[1] for r in leituras)
                    placa_txt, tipo = validar_placa(txt)
                    if placa_txt:
                        self.historico.append(placa_txt)

            # Placa estável (votação)
            if self.historico:
                placa_txt, _ = Counter(self.historico).most_common(1)[0]
                placa_txt, tipo = validar_placa(placa_txt)

            if fazer_ocr:
                self._ultimo_ocr = agora

            resultado.append((x1, y1, x2, y2, placa_txt, tipo, conf))

        return resultado

    def _detectar_sem_modelo(self, frame: np.ndarray):
        """Heurística simples: retângulos brancos/amarelos no terço inferior."""
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.55):h, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Branco (Mercosul) e amarelo (antiga)
        mask_branco  = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
        mask_amarelo = cv2.inRange(hsv, (20, 80, 120), (35, 255, 255))
        mask = cv2.bitwise_or(mask_branco, mask_amarelo)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regioes = []
        for cnt in contornos:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = cw / max(ch, 1)
            if 100 < cw < 500 and 20 < ch < 120 and 2.0 < ratio < 6.0:
                y_abs = y + int(h * 0.55)
                regioes.append((x, y_abs, x + cw, y_abs + ch, 0.6))
        return regioes


# ═══════════════════════════════════════════════════════════════════════════════
# ANALISADOR DE BASCULANTE
# ═══════════════════════════════════════════════════════════════════════════════
class AnalisadorBasculante:
    """
    Analisa se o basculante está limpo ou sujo usando visão computacional:
    - Variância de textura (superfície suja = mais irregular)
    - Índice de escurecimento (terra/lama = tons escuros e marrons)
    - Saliência de bordas (resíduos criam bordas extras)
    """

    LIMIAR_SUJO = 55.0   # Score acima disso = SUJO

    def __init__(self, modelo_path: str | None = None):
        self.yolo = None
        if modelo_path and YOLO_OK:
            p = Path(modelo_path)
            if p.exists():
                print(f"[INFO] Carregando YOLO basculante: {modelo_path}")
                self.yolo = YOLO(str(p))

        self.historico: deque[tuple] = deque(maxlen=30)
        self._ultimo_score: float = 0.0
        self._ultimo_status: str = "Aguardando..."

    def analisar(self, frame: np.ndarray):
        """Retorna (status, score, roi_coords)."""
        roi, coords = self._extrair_roi(frame)

        if roi is None or roi.size == 0:
            return self._ultimo_status, self._ultimo_score, None

        score = self._calcular_score_sujeira(roi)

        self.historico.append(score)
        score_medio = float(np.mean(list(self.historico)))

        if score_medio >= self.LIMIAR_SUJO:
            status = "SUJO"
        elif score_medio >= self.LIMIAR_SUJO * 0.65:
            status = "LEVEMENTE SUJO"
        else:
            status = "LIMPO"

        self._ultimo_score  = score_medio
        self._ultimo_status = status
        return status, score_medio, coords

    def _extrair_roi(self, frame: np.ndarray):
        """Extrai a região do basculante. Sem modelo: usa zona central-superior."""
        h, w = frame.shape[:2]
        # Zona provável do basculante: 15%–80% da altura, centro horizontal
        y1 = int(h * 0.10)
        y2 = int(h * 0.82)
        x1 = int(w * 0.08)
        x2 = int(w * 0.92)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _calcular_score_sujeira(self, roi: np.ndarray) -> float:
        """Score 0–100. Quanto maior, mais sujo."""
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. VARIÂNCIA DE TEXTURA — superfície suja é mais irregular
        laplacian  = cv2.Laplacian(cinza, cv2.CV_64F)
        var_textura = np.var(laplacian)
        score_textura = min(np.log1p(var_textura) * 6, 100)

        # 2. ESCURECIMENTO — terra/lama são escuros (V baixo no HSV)
        v_canal = hsv[:, :, 2].astype(float)
        escuro_pct = np.mean(v_canal < 90) * 100
        score_escuro = min(escuro_pct * 1.5, 100)

        # 3. SATURAÇÃO MARROM — terra tem saturação média e hue amarelo-laranja
        h_canal = hsv[:, :, 0].astype(float)
        s_canal = hsv[:, :, 1].astype(float)
        mascara_marrom = (
            (h_canal >= 8) & (h_canal <= 25) &
            (s_canal >= 50) & (s_canal <= 200)
        )
        marrom_pct = np.mean(mascara_marrom) * 100
        score_marrom = min(marrom_pct * 2.5, 100)

        # 4. HETEROGENEIDADE DE COR — superfície limpa tem cor uniforme
        std_h = float(np.std(h_canal))
        std_s = float(np.std(s_canal))
        score_heter = min((std_h + std_s) / 2, 100)

        # Ponderação final
        score = (
            score_textura * 0.30 +
            score_escuro  * 0.25 +
            score_marrom  * 0.30 +
            score_heter   * 0.15
        )
        return float(np.clip(score, 0, 100))


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERIZADOR DO DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
FONTE      = cv2.FONT_HERSHEY_SIMPLEX
MONO       = cv2.FONT_HERSHEY_SIMPLEX

COR_VERDE  = (80, 220,  80)
COR_AMARELO= (40, 210, 255)
COR_VERMELHO=(50,  50, 220)
COR_BRANCO = (240,240,240)
COR_CINZA  = (130,130,130)
COR_FUNDO  = ( 18, 18, 22)
COR_PAINEL = ( 28, 28, 35)
COR_BORDA  = ( 55, 55, 70)


def cor_status(status: str):
    s = status.upper()
    if "SUJO" in s and "LEVE" not in s:
        return COR_VERMELHO
    if "LEVE" in s:
        return COR_AMARELO
    if "LIMPO" in s:
        return COR_VERDE
    return COR_CINZA


def barra_progresso(canvas, x, y, w, h, valor, cor):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 60), -1)
    preench = int(w * min(valor, 100) / 100)
    if preench > 0:
        cv2.rectangle(canvas, (x, y), (x + preench, y + h), cor, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COR_BORDA, 1)


def texto(canvas, txt, x, y, escala=0.55, cor=COR_BRANCO, espessura=1):
    cv2.putText(canvas, txt, (x, y), FONTE, escala, cor, espessura, cv2.LINE_AA)


def renderizar_frame_camera(frame, titulo, largura, altura):
    """Redimensiona e adiciona borda/título ao frame da câmera."""
    f = cv2.resize(frame, (largura, altura))
    overlay = f.copy()
    cv2.rectangle(overlay, (0, 0), (largura, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, f, 0.4, 0, f)
    texto(f, titulo, 8, 20, 0.6, COR_BRANCO, 1)
    cv2.rectangle(f, (0, 0), (largura - 1, altura - 1), COR_BORDA, 2)
    return f


def renderizar_dashboard(
    largura, altura,
    placa_atual, tipo_placa,
    status_bascul, score_bascul,
    hist_entradas, hist_placas, hist_bascul,
    fps_placa, fps_bascul,
    logo_img=None,
):
    dash = np.full((altura, largura, 3), COR_FUNDO, dtype=np.uint8)

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    cv2.rectangle(dash, (0, 0), (largura, 52), COR_PAINEL, -1)
    cv2.rectangle(dash, (0, 52), (largura, 53), COR_BORDA, -1)
    texto(dash, "MONITORAMENTO DE CAMINHOES", 12, 22, 0.48, COR_BRANCO, 2)
    texto(dash, datetime.now().strftime("%d/%m/%Y  %H:%M:%S"),
          12, 44, 0.45, COR_CINZA)

    # Inserir a logo no canto superior direito do painel
    if logo_img is not None:
        lh, lw = logo_img.shape[:2]
        y_offset = 6
        x_offset = largura - lw - 8
        roi = dash[y_offset:y_offset+lh, x_offset:x_offset+lw]
        cv2.copyTo(logo_img, None, roi)

    y = 65

    # ── Placa atual ─────────────────────────────────────────────────────────
    cv2.rectangle(dash, (8, y), (largura - 8, y + 72), COR_PAINEL, -1)
    cv2.rectangle(dash, (8, y), (largura - 8, y + 72), COR_BORDA, 1)
    texto(dash, "PLACA DETECTADA", 16, y + 17, 0.50, COR_CINZA)
    placa_disp = placa_atual or "---"
    texto(dash, placa_disp, 16, y + 52, 1.1, COR_BRANCO, 2)
    if tipo_placa:
        texto(dash, tipo_placa, largura - 100, y + 52, 0.5, COR_AMARELO)
    y += 82

    # ── Status basculante ───────────────────────────────────────────────────
    cv2.rectangle(dash, (8, y), (largura - 8, y + 80), COR_PAINEL, -1)
    cv2.rectangle(dash, (8, y), (largura - 8, y + 80), COR_BORDA, 1)
    texto(dash, "BASCULANTE", 16, y + 17, 0.50, COR_CINZA)
    cor_b = cor_status(status_bascul)
    texto(dash, status_bascul, 16, y + 46, 0.85, cor_b, 2)
    texto(dash, f"Score: {score_bascul:.1f}/100", 16, y + 68, 0.45, COR_CINZA)
    barra_progresso(dash, largura // 2, y + 30, largura // 2 - 16, 14,
                    score_bascul, cor_b)
    y += 90

    # ── Divisor ─────────────────────────────────────────────────────────────
    cv2.line(dash, (8, y), (largura - 8, y), COR_BORDA, 1)
    y += 10

    # ── Histórico de entradas ───────────────────────────────────────────────
    texto(dash, "HISTORICO DE ENTRADAS", 12, y + 14, 0.48, COR_CINZA)
    y += 22
    for row in hist_entradas[:6]:
        ts, placa, st_b = row
        hora = ts[11:16] if len(ts) > 10 else ts
        cor_linha = cor_status(st_b or "")
        cv2.rectangle(dash, (8, y), (largura - 8, y + 22), (25, 25, 32), -1)
        texto(dash, hora,  14, y + 15, 0.42, COR_CINZA)
        texto(dash, placa or "???", 70, y + 15, 0.45, COR_BRANCO)
        texto(dash, (st_b or "")[:8], largura - 95, y + 15, 0.40, cor_linha)
        y += 24

    y += 6
    cv2.line(dash, (8, y), (largura - 8, y), COR_BORDA, 1)
    y += 10

    # ── Histórico de placas ─────────────────────────────────────────────────
    texto(dash, "ULTIMAS PLACAS LIDAS", 12, y + 14, 0.48, COR_CINZA)
    y += 22
    for row in hist_placas[:5]:
        ts, placa, tipo, conf = row
        hora = ts[11:16] if len(ts) > 10 else ts
        cv2.rectangle(dash, (8, y), (largura - 8, y + 20), (25, 25, 32), -1)
        texto(dash, hora,  14, y + 14, 0.40, COR_CINZA)
        texto(dash, placa, 70, y + 14, 0.44, COR_BRANCO)
        texto(dash, f"{(conf or 0)*100:.0f}%", largura - 48, y + 14, 0.40, COR_CINZA)
        y += 22

    y += 6
    cv2.line(dash, (8, y), (largura - 8, y), COR_BORDA, 1)
    y += 10

    # ── Histórico de basculantes ─────────────────────────────────────────────
    texto(dash, "ULTIMOS BASCULANTES", 12, y + 14, 0.48, COR_CINZA)
    y += 22
    for row in hist_bascul[:5]:
        ts, status, score, placa_ref = row
        hora = ts[11:16] if len(ts) > 10 else ts
        cor_b2 = cor_status(status or "")
        cv2.rectangle(dash, (8, y), (largura - 8, y + 20), (25, 25, 32), -1)
        texto(dash, hora,         14, y + 14, 0.40, COR_CINZA)
        texto(dash, (status or "")[:12], 70, y + 14, 0.42, cor_b2)
        texto(dash, f"{(score or 0):.0f}", largura - 38, y + 14, 0.40, COR_CINZA)
        y += 22

    # ── FPS ─────────────────────────────────────────────────────────────────
    cv2.rectangle(dash, (0, altura - 26), (largura, altura), COR_PAINEL, -1)
    texto(dash, f"CAM1 {fps_placa:.1f}fps  CAM2 {fps_bascul:.1f}fps",
          10, altura - 8, 0.42, COR_CINZA)
    texto(dash, "Q/ESC = sair", largura - 110, altura - 8, 0.40, COR_CINZA)

    return dash


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# BARRA DE PROGRESSO DO VÍDEO
# ═══════════════════════════════════════════════════════════════════════════════
def renderizar_barra_progresso(canvas, frame_atual, total_frames, pausado, velocidade):
    h, w = canvas.shape[:2]
    barra_y = h - 28
    barra_h = 6
    barra_x1, barra_x2 = 10, w - 10

    # Fundo da barra
    cv2.rectangle(canvas, (barra_x1, barra_y), (barra_x2, barra_y + barra_h), (50, 50, 60), -1)

    # Preenchimento proporcional
    if total_frames > 0:
        progresso = int((barra_x2 - barra_x1) * frame_atual / total_frames)
        cv2.rectangle(canvas, (barra_x1, barra_y),
                      (barra_x1 + progresso, barra_y + barra_h), (80, 180, 255), -1)
        # Bolinha indicadora
        cx = barra_x1 + progresso
        cv2.circle(canvas, (cx, barra_y + barra_h // 2), 7, (120, 200, 255), -1)

    # Tempo atual / total
    fps_vid = 30
    seg_atual = int(frame_atual / fps_vid)
    seg_total = int(total_frames / fps_vid)
    tempo_str = f"{seg_atual//60:02d}:{seg_atual%60:02d} / {seg_total//60:02d}:{seg_total%60:02d}"
    cv2.putText(canvas, tempo_str, (barra_x1, barra_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)

    # Status de reprodução
    estado = "⏸ PAUSADO" if pausado else f"▶ {velocidade:.1f}x"
    cv2.putText(canvas, estado, (barra_x2 - 90, barra_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)

    # Dica de controles
    cv2.putText(canvas, "ESPACO=pausar  ←→=5s  +/-=vel  S=salvar  Q=sair",
                (barra_x1, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 90), 1, cv2.LINE_AA)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# PEDIR ARQUIVO AO USUÁRIO SE NÃO FOR PASSADO POR ARGUMENTO
# ═══════════════════════════════════════════════════════════════════════════════
def pedir_video(descricao: str) -> str:
    while True:
        print(f"\n{'─'*50}")
        print(f"  Informe o caminho do vídeo de {descricao}:")
        print(f"  (ex: C:\\videos\\placa.mp4  ou  /home/user/placa.mp4)")
        caminho = input("  >> ").strip().strip('"').strip("'")
        if Path(caminho).exists():
            return caminho
        print(f"  [ERRO] Arquivo não encontrado: {caminho}")
        print("  Verifique o caminho e tente novamente.")


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
def executar(args):
    # ── Resolver caminhos dos vídeos ────────────────────────────────────────
    path_placa = args.video_placa or pedir_video("PLACA (câmera 1)")
    path_bascul = args.video_basculante or pedir_video("BASCULANTE (câmera 2)")

    for label, p in [("Placa", path_placa), ("Basculante", path_bascul)]:
        if not Path(p).exists():
            print(f"[ERRO] Vídeo de {label} não encontrado: {p}")
            sys.exit(1)

    print(f"\n[INFO] Vídeo de placa:      {path_placa}")
    print(f"[INFO] Vídeo de basculante: {path_bascul}")

    banco = Banco("registros")
    Path("registros/placas").mkdir(parents=True, exist_ok=True)
    Path("registros/basculantes").mkdir(parents=True, exist_ok=True)

    det_placa  = DetectorPlaca(args.modelo_placa)
    det_bascul = AnalisadorBasculante(args.modelo_basculante)

    # ── Carregar Logo Oficial Granorte ──────────────────────────────────────
    logo_img = cv2.imread("logo_granorte.png")
    if logo_img is not None:
        largura_logo = 65
        proporcao = largura_logo / logo_img.shape[1]
        altura_logo = int(logo_img.shape[0] * proporcao)
        logo_img = cv2.resize(logo_img, (largura_logo, altura_logo))
    else:
        print("[AVISO] Arquivo 'logo_granorte.png' nao localizado. Interface carregara sem a logo.")

    cap1 = cv2.VideoCapture(path_placa)
    cap2 = cv2.VideoCapture(path_bascul)

    if not cap1.isOpened():
        print(f"[ERRO] Não foi possível abrir: {path_placa}")
        sys.exit(1)
    if not cap2.isOpened():
        print(f"[ERRO] Não foi possível abrir: {path_bascul}")
        sys.exit(1)

    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid1 = cap1.get(cv2.CAP_PROP_FPS) or 30
    fps_vid2 = cap2.get(cv2.CAP_PROP_FPS) or 30

    print(f"[INFO] Vídeo 1: {total1} frames @ {fps_vid1:.1f} FPS")
    print(f"[INFO] Vídeo 2: {total2} frames @ {fps_vid2:.1f} FPS")

    # Dimensões do layout
    CAM_W, CAM_H = 640, 400
    DASH_W       = 310
    TOTAL_H      = CAM_H * 2

    # Writer de saída (opcional)
    writer = None
    if args.salvar_saida:
        nome_saida = args.arquivo_saida or "saida_monitoramento.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(nome_saida, fourcc, fps_vid1, (CAM_W + DASH_W, TOTAL_H))
        print(f"[INFO] Salvando resultado em: {nome_saida}")

    # Estado de reprodução
    pausado    = False
    velocidade = 1.0      # multiplicador: 0.5x, 1x, 2x, 4x
    frame_n1   = 0
    frame_n2   = 0

    # Estado de detecção
    placa_atual   = None
    tipo_placa    = None
    status_bascul = "Aguardando..."
    score_bascul  = 0.0
    fps1 = fps2  = 0.0
    t1 = t2      = time.perf_counter()

    ultima_placa_salva  = None
    ultimo_bascul_salvo = None
    t_ultimo_bascul     = 0.0

    # Cache do último frame (para quando está pausado)
    ultimo_frame1 = None
    ultimo_frame2 = None

    print("\n[INFO] Reprodução iniciada — ESPAÇO para pausar, Q para sair\n")

    while True:
        # ── Controle de velocidade: delay entre frames ───────────────────────
        delay_ms = max(1, int(1000 / (fps_vid1 * velocidade)))

        if not pausado:
            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()

            # Loop automático ao chegar no fim
            if not ok1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_n1 = 0
                ok1, frame1 = cap1.read()
            if not ok2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_n2 = 0
                ok2, frame2 = cap2.read()

            if not ok1 or not ok2:
                print("[INFO] Fim do vídeo.")
                break

            frame_n1 = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
            frame_n2 = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))
            ultimo_frame1 = frame1.copy()
            ultimo_frame2 = frame2.copy()

            # ── Vídeo 1: Placa ───────────────────────────────────────────────
            deteccoes = det_placa.detectar(frame1, args.ocr_intervalo)
            for (x1, y1, x2, y2, placa, tipo, conf) in deteccoes:
                cor = COR_VERDE if placa else COR_CINZA
                cv2.rectangle(frame1, (x1, y1), (x2, y2), cor, 3)
                label = placa or "Lendo..."
                (tw, th), _ = cv2.getTextSize(label, FONTE, 0.7, 2)
                cv2.rectangle(frame1, (x1, y1 - th - 10), (x1 + tw + 6, y1), cor, -1)
                cv2.putText(frame1, label, (x1 + 3, y1 - 4),
                            FONTE, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                if placa:
                    placa_atual = placa
                    tipo_placa  = tipo
                    if placa != ultima_placa_salva:
                        ultima_placa_salva = placa
                        foto_path = f"registros/placas/{placa}_{int(time.time())}.jpg"
                        cv2.imwrite(foto_path, frame1)
                        banco.salvar_placa(placa, tipo, conf, foto_path)
                        banco.registrar_entrada(placa, status_bascul)
                        print(f"  🚛  Placa: {placa} ({tipo}) | Basculante: {status_bascul}")

            agora1 = time.perf_counter()
            fps1   = 1.0 / max(agora1 - t1, 1e-6)
            t1     = agora1

            # ── Vídeo 2: Basculante ──────────────────────────────────────────
            status_bascul, score_bascul, coords = det_bascul.analisar(frame2)

            if coords:
                bx1, by1, bx2, by2 = coords
                cv2.rectangle(frame2, (bx1, by1), (bx2, by2), cor_status(status_bascul), 3)

            h2, w2 = frame2.shape[:2]
            overlay2 = frame2.copy()
            cv2.rectangle(overlay2, (0, h2 - 55), (w2, h2), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.65, frame2, 0.35, 0, frame2)
            cv2.putText(frame2, status_bascul, (12, h2 - 28),
                        FONTE, 0.9, cor_status(status_bascul), 2, cv2.LINE_AA)
            cv2.putText(frame2, f"Score: {score_bascul:.1f}/100", (12, h2 - 8),
                        FONTE, 0.5, COR_CINZA, 1, cv2.LINE_AA)

            agora_t = time.time()
            if status_bascul != ultimo_bascul_salvo and (agora_t - t_ultimo_bascul) > 5:
                foto_b = f"registros/basculantes/bascul_{int(agora_t)}.jpg"
                cv2.imwrite(foto_b, frame2)
                banco.salvar_basculante(status_bascul, score_bascul,
                                        placa_atual or "", foto_b)
                ultimo_bascul_salvo = status_bascul
                t_ultimo_bascul     = agora_t
                print(f"  🪣  Basculante: {status_bascul} (score={score_bascul:.1f})")

            agora2 = time.perf_counter()
            fps2   = 1.0 / max(agora2 - t2, 1e-6)
            t2     = agora2

        else:
            # Pausado: reusar último frame processado
            frame1 = ultimo_frame1.copy() if ultimo_frame1 is not None else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            frame2 = ultimo_frame2.copy() if ultimo_frame2 is not None else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

        # ── Montar tela ──────────────────────────────────────────────────────
        f1 = renderizar_frame_camera(frame1, "VID 1 — PLACA", CAM_W, CAM_H)
        f2 = renderizar_frame_camera(frame2, "VID 2 — BASCULANTE", CAM_W, CAM_H)

        # Barra de progresso em cada frame
        renderizar_barra_progresso(f1, frame_n1, total1, pausado, velocidade)
        renderizar_barra_progresso(f2, frame_n2, total2, pausado, velocidade)

        coluna_cam = np.vstack([f1, f2])

        dash = renderizar_dashboard(
            DASH_W, TOTAL_H,
            placa_atual, tipo_placa,
            status_bascul, score_bascul,
            banco.historico_entradas(6),
            banco.historico_placas(5),
            banco.historico_basculantes(5),
            fps1, fps2,
            logo_img,
        )

        tela = np.hstack([coluna_cam, dash])

        if writer:
            writer.write(tela)

        cv2.imshow("Sistema de Monitoramento de Caminhoes", tela)

        # ── Teclas ───────────────────────────────────────────────────────────
        tecla = cv2.waitKey(delay_ms) & 0xFF

        if tecla in (ord("q"), ord("Q"), 27):
            print("[INFO] Encerrado pelo usuário.")
            break

        elif tecla == ord(" "):                          # ESPAÇO — pausar/retomar
            pausado = not pausado
            print(f"[INFO] {'Pausado' if pausado else 'Retomado'}")

        elif tecla == 83 or tecla == ord("d"):           # → — avançar 5s
            pulos = int(fps_vid1 * 5)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, min(frame_n1 + pulos, total1 - 1))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, min(frame_n2 + pulos, total2 - 1))

        elif tecla == 81 or tecla == ord("a"):           # ← — voltar 5s
            pulos = int(fps_vid1 * 5)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, max(frame_n1 - pulos, 0))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, max(frame_n2 - pulos, 0))

        elif tecla in (ord("+"), ord("=")):              # + — mais rápido
            velocidade = min(velocidade * 2, 8.0)
            print(f"[INFO] Velocidade: {velocidade:.1f}x")

        elif tecla == ord("-"):                          # - — mais lento
            velocidade = max(velocidade / 2, 0.25)
            print(f"[INFO] Velocidade: {velocidade:.1f}x")

        elif tecla in (ord("s"), ord("S")):              # S — salvar frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_frame = f"registros/frame_manual_{ts}.jpg"
            cv2.imwrite(nome_frame, tela)
            print(f"[INFO] Frame salvo: {nome_frame}")

    cap1.release()
    cap2.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n[RESUMO FINAL]")
    entradas = banco.historico_entradas(100)
    print(f"  Total de entradas registradas: {len(entradas)}")
    print("  Sistema encerrado.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENTOS
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Monitoramento de caminhões por vídeo — placa + basculante"
    )
    p.add_argument("--video-placa",
                   default=None, dest="video_placa",
                   help="Caminho do vídeo da PLACA (se omitido, será pedido no terminal)")
    p.add_argument("--video-basculante",
                   default=None, dest="video_basculante",
                   help="Caminho do vídeo do BASCULANTE (se omitido, será pedido no terminal)")
    p.add_argument("--modelo-placa",
                   default=None, dest="modelo_placa",
                   help="Modelo .pt YOLO para placa (opcional — melhora a detecção)")
    p.add_argument("--modelo-basculante",
                   default=None, dest="modelo_basculante",
                   help="Modelo .pt YOLO para basculante (opcional)")
    p.add_argument("--ocr-intervalo",
                   type=float, default=1.5, dest="ocr_intervalo",
                   help="Segundos entre leituras OCR (padrão: 1.5)")
    p.add_argument("--salvar-saida",
                   action="store_true", dest="salvar_saida",
                   help="Salvar vídeo de saída com as detecções")
    p.add_argument("--arquivo-saida",
                   default=None, dest="arquivo_saida",
                   help="Nome do vídeo de saída (padrão: saida_monitoramento.mp4)")
    return p.parse_args()


if __name__ == "__main__":
    executar(parse_args())
