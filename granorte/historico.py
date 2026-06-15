"""
Visualizador de histórico — lê o banco SQLite gerado pelo detector.py
e exibe uma tela de relatório com filtros por data e placa.

USO:
    python historico.py
    python historico.py --data 2024-12-01
    python historico.py --placa ABC1234
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

FONTE      = cv2.FONT_HERSHEY_SIMPLEX
COR_FUNDO  = ( 14,  14,  18)
COR_PAINEL = ( 24,  24,  30)
COR_BORDA  = ( 50,  50,  65)
COR_BRANCO = (235, 235, 235)
COR_CINZA  = (120, 120, 130)
COR_VERDE  = ( 70, 210,  70)
COR_AMAREL = ( 40, 200, 255)
COR_VERM   = ( 50,  50, 210)
COR_AZUL   = (200, 150,  50)

W, H = 1000, 700


def cor_status(s: str):
    s = (s or "").upper()
    if "SUJO" in s and "LEVE" not in s:
        return COR_VERM
    if "LEVE" in s:
        return COR_AMAREL
    if "LIMPO" in s:
        return COR_VERDE
    return COR_CINZA


def txt(canvas, t, x, y, esc=0.5, cor=COR_BRANCO, esp=1):
    cv2.putText(canvas, str(t), (x, y), FONTE, esc, cor, esp, cv2.LINE_AA)


def linha_h(canvas, y, x1=8, x2=W - 8, cor=COR_BORDA):
    cv2.line(canvas, (x1, y), (x2, y), cor, 1)


def carregar_dados(db_path, filtro_data=None, filtro_placa=None):
    conn = sqlite3.connect(str(db_path))

    # Entradas
    q = "SELECT timestamp,placa,status_basculante,observacao FROM entradas WHERE 1=1"
    params = []
    if filtro_data:
        q += " AND timestamp LIKE ?"
        params.append(f"{filtro_data}%")
    if filtro_placa:
        q += " AND placa LIKE ?"
        params.append(f"%{filtro_placa}%")
    q += " ORDER BY id DESC LIMIT 200"
    entradas = conn.execute(q, params).fetchall()

    # Resumo de placas únicas
    placas_unicas = conn.execute(
        "SELECT placa, COUNT(*) as n, MAX(timestamp) FROM placas "
        "GROUP BY placa ORDER BY n DESC LIMIT 30"
    ).fetchall()

    # Estatísticas de basculante
    stats_b = conn.execute(
        "SELECT status, COUNT(*) FROM basculantes GROUP BY status"
    ).fetchall()

    conn.close()
    return entradas, placas_unicas, stats_b


def pagina_principal(entradas, placas_unicas, stats_b, filtro_data, filtro_placa, pagina, total_pags):
    canvas = np.full((H, W, 3), COR_FUNDO, dtype=np.uint8)

    # Cabeçalho
    cv2.rectangle(canvas, (0, 0), (W, 50), COR_PAINEL, -1)
    linha_h(canvas, 50)
    txt(canvas, "HISTORICO — MONITORAMENTO DE CAMINHOES", 14, 22, 0.72, COR_BRANCO, 2)
    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    txt(canvas, ts, W - 200, 22, 0.46, COR_CINZA)

    y = 60

    # Filtros ativos
    filtro_str = f"Data: {filtro_data or 'todas'}   Placa: {filtro_placa or 'todas'}"
    txt(canvas, filtro_str, 14, y + 18, 0.48, COR_CINZA)
    txt(canvas, f"Pagina {pagina+1}/{max(total_pags,1)}", W - 140, y + 18, 0.46, COR_CINZA)
    y += 30

    linha_h(canvas, y)
    y += 8

    # ── Tabela de entradas ──────────────────────────────────────────────────
    # Cabeçalho da tabela
    cv2.rectangle(canvas, (8, y), (W - 8, y + 26), COR_PAINEL, -1)
    txt(canvas, "DATA/HORA",         16, y + 18, 0.48, COR_CINZA)
    txt(canvas, "PLACA",            180, y + 18, 0.48, COR_CINZA)
    txt(canvas, "BASCULANTE",       340, y + 18, 0.48, COR_CINZA)
    txt(canvas, "OBSERVACAO",       540, y + 18, 0.48, COR_CINZA)
    y += 28

    por_pag = 12
    inicio  = pagina * por_pag
    fatia   = entradas[inicio: inicio + por_pag]

    for i, row in enumerate(fatia):
        ts_r, placa, status_b, obs = row
        hora = ts_r[:16] if ts_r else "—"
        bg = (22, 22, 28) if i % 2 == 0 else (28, 28, 36)
        cv2.rectangle(canvas, (8, y), (W - 8, y + 24), bg, -1)

        txt(canvas, hora,                 16, y + 16, 0.44, COR_CINZA)
        txt(canvas, placa or "???",      180, y + 16, 0.48, COR_BRANCO)
        txt(canvas, (status_b or "—")[:16], 340, y + 16, 0.46, cor_status(status_b))
        txt(canvas, (obs or "")[:35],    540, y + 16, 0.40, COR_CINZA)
        y += 24

    y += 10
    linha_h(canvas, y)
    y += 10

    # ── Painéis inferiores ──────────────────────────────────────────────────
    painel_w = (W - 24) // 2

    # Painel esquerdo: Placas únicas
    px, py = 8, y
    cv2.rectangle(canvas, (px, py), (px + painel_w, H - 40), COR_PAINEL, -1)
    cv2.rectangle(canvas, (px, py), (px + painel_w, H - 40), COR_BORDA, 1)
    txt(canvas, "PLACAS MAIS FREQUENTES", px + 10, py + 18, 0.50, COR_CINZA)
    ry = py + 30
    for placa, n, ultima in placas_unicas[:8]:
        txt(canvas, placa,          px + 10, ry + 15, 0.52, COR_BRANCO)
        txt(canvas, f"{n}x",        px + 120, ry + 15, 0.46, COR_AMAREL)
        ultima_h = ultima[11:16] if ultima and len(ultima) > 10 else ""
        txt(canvas, ultima_h,       px + 160, ry + 15, 0.40, COR_CINZA)
        ry += 22

    # Painel direito: Stats basculante
    px2 = 8 + painel_w + 8
    cv2.rectangle(canvas, (px2, py), (W - 8, H - 40), COR_PAINEL, -1)
    cv2.rectangle(canvas, (px2, py), (W - 8, H - 40), COR_BORDA, 1)
    txt(canvas, "STATUS DO BASCULANTE", px2 + 10, py + 18, 0.50, COR_CINZA)

    total_b = sum(n for _, n in stats_b) or 1
    ry2 = py + 30
    for status, n in stats_b:
        pct = n / total_b * 100
        txt(canvas, (status or "—")[:18], px2 + 10, ry2 + 15, 0.50, cor_status(status))
        txt(canvas, f"{n}x  ({pct:.0f}%)", px2 + 180, ry2 + 15, 0.44, COR_CINZA)
        barra_x = px2 + 10
        barra_w = painel_w - 20
        cv2.rectangle(canvas, (barra_x, ry2 + 18), (barra_x + barra_w, ry2 + 26), (40, 40, 50), -1)
        fill = int(barra_w * pct / 100)
        if fill > 0:
            cv2.rectangle(canvas, (barra_x, ry2 + 18), (barra_x + fill, ry2 + 26), cor_status(status), -1)
        ry2 += 36

    # Rodapé
    cv2.rectangle(canvas, (0, H - 38), (W, H), COR_PAINEL, -1)
    linha_h(canvas, H - 38)
    txt(canvas, "SETAS: paginar   Q/ESC: sair   R: atualizar",
        14, H - 14, 0.44, COR_CINZA)
    txt(canvas, f"Total de entradas: {len(entradas)}",
        W - 220, H - 14, 0.44, COR_CINZA)

    return canvas


def executar(args):
    db = Path("registros/historico.db")
    if not db.exists():
        print(f"[ERRO] Banco não encontrado: {db}")
        print("       Execute primeiro o detector.py para gerar dados.")
        return

    pagina    = 0
    por_pagina= 12

    def recarregar():
        e, p, s = carregar_dados(db, args.data, args.placa)
        total   = max(1, (len(e) + por_pagina - 1) // por_pagina)
        return e, p, s, total

    entradas, placas_u, stats_b, total_pags = recarregar()

    print("[INFO] Visualizador de histórico — Q/ESC para sair")

    while True:
        canvas = pagina_principal(
            entradas, placas_u, stats_b,
            args.data, args.placa,
            pagina, total_pags,
        )
        cv2.imshow("Historico — Monitoramento de Caminhoes", canvas)

        tecla = cv2.waitKey(500) & 0xFF

        if tecla in (ord("q"), ord("Q"), 27):
            break
        elif tecla in (81, ord("a"), ord("A")):   # seta esquerda / A
            pagina = max(0, pagina - 1)
        elif tecla in (83, ord("d"), ord("D")):   # seta direita / D
            pagina = min(total_pags - 1, pagina + 1)
        elif tecla in (ord("r"), ord("R")):
            entradas, placas_u, stats_b, total_pags = recarregar()
            pagina = 0
            print("[INFO] Dados atualizados.")

    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Visualizador de histórico de caminhões")
    p.add_argument("--data",  default=None, help="Filtrar por data (ex: 2024-12-01)")
    p.add_argument("--placa", default=None, help="Filtrar por placa (parcial)")
    return p.parse_args()


if __name__ == "__main__":
    executar(parse_args())
