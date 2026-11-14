#!/usr/bin/env python3
"""Render the concept-2 loop as a GitHub-compatible animated GIF.

Run from any directory: python3 assets/readme/concepts/concept-2/render.py
Requires Pillow. Timeline is 8 seconds, 48 frames at 6 fps; all motion is periodic.
"""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1000, 470
FRAMES, FPS = 48, 6
BG = (8, 13, 22)
PANEL = (14, 22, 34)
EDGE = (34, 49, 68)
MUTED = (131, 151, 175)
TEXT = (229, 239, 250)
CYAN = (65, 217, 227)
BLUE = (87, 150, 255)
VIOLET = (176, 130, 255)
GREEN = (105, 231, 174)
AMBER = (255, 191, 93)


def font(size, bold=False):
    candidates = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                  '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    for p in candidates:
        if Path(p).exists(): return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_TITLE, F_HEAD, F_BODY, F_SMALL, F_MONO = font(23, True), font(15, True), font(13), font(11), font(11)

def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def centered(draw, xy, text, fnt, fill):
    draw.text(xy, text, font=fnt, fill=fill, anchor='mm')

def frame(i):
    t = i / FRAMES
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    # restrained blueprint grid
    for x in range(28, W, 32): d.line((x, 0, x, H), fill=(13, 21, 33), width=1)
    for y in range(10, H, 32): d.line((0, y, W, y), fill=(13, 21, 33), width=1)
    d.text((34, 25), 'ROUTING / 02', font=F_SMALL, fill=CYAN)
    d.text((34, 48), 'От резюме к подходящим вакансиям', font=F_TITLE, fill=TEXT)
    d.text((966, 34), 'RETRIEVAL PIPELINE', font=F_SMALL, fill=MUTED, anchor='ra')
    d.line((34, 88, 966, 88), fill=EDGE, width=1)

    # Three-stage deterministic pipeline, with editorial notes and explicit shortlist.
    boxes = [(34, 142, 250, 338), (286, 142, 535, 338), (571, 142, 809, 338), (845, 166, 966, 314)]
    for b in boxes: rounded(d, b, 13, PANEL, EDGE, 1)
    # connector rails behind all moving packets
    rails = [(250, 240, 286, 240, CYAN), (535, 240, 571, 240, BLUE), (809, 240, 845, 240, VIOLET)]
    for x1,y1,x2,y2,c in rails:
        d.line((x1,y1,x2,y2), fill=(33,51,70), width=3)
        d.ellipse((x1-3,y1-3,x1+3,y1+3), fill=c)
        d.ellipse((x2-3,y2-3,x2+3,y2+3), fill=c)

    # Stage 1: hard rule gate
    d.text((52, 160), '01  /  GATE', font=F_SMALL, fill=CYAN)
    d.text((52, 184), 'Детерминированный', font=F_HEAD, fill=TEXT)
    d.text((52, 204), 'rule gate', font=F_HEAD, fill=TEXT)
    d.text((52, 235), 'pass / fail / unknown', font=F_BODY, fill=MUTED)
    for j, (label, color, fill) in enumerate([('pass · Python', GREEN, (18,37,42)), ('unknown · keep', AMBER, (41,37,29)), ('fail · exclude', (236,120,120), (43,29,33))]):
        y=264+j*21
        rounded(d, (52,y,230,y+17), 5, fill, None)
        d.ellipse((60,y+5,66,y+11), fill=color)
        d.text((73,y+2), label, font=F_SMALL, fill=color)
    d.text((52, 345), 'unknown included by default', font=F_SMALL, fill=MUTED)

    # Stage 2: shortlist
    d.text((304, 160), '02  /  RETRIEVE', font=F_SMALL, fill=BLUE)
    d.text((304, 184), 'BM25Plus shortlist', font=F_HEAD, fill=TEXT)
    d.text((304, 211), 'Лексический поиск → shortlist', font=F_BODY, fill=MUTED)
    for j,(name,score) in enumerate([('Frontend', 'matched'),('Backend', 'matched'),('Fullstack', 'matched')]):
        y=244+j*25
        rounded(d, (304,y,517,y+20), 5, (18,29,47), None)
        d.text((313,y+3), name, font=F_SMALL, fill=(195,214,241))
        d.text((507,y+3), score, font=F_SMALL, fill=BLUE, anchor='ra')

    # Stage 3: semantic reranking
    d.text((589, 160), '03  /  RERANK', font=F_SMALL, fill=VIOLET)
    d.text((589, 184), 'EmbeddingGemma 2', font=F_HEAD, fill=TEXT)
    d.text((589, 211), 'Ранжирование shortlist', font=F_BODY, fill=MUTED)
    for j,(name,wid,col) in enumerate([('Backend',158,VIOLET),('Frontend',117,BLUE),('Fullstack',79,(83,101,128))]):
        y=249+j*25
        d.text((589,y), name, font=F_SMALL, fill=(197,210,227))
        rounded(d, (713,y+2,789,y+10), 4, (31,42,58), None)
        rounded(d, (713,y+2,713+wid*0.45,y+10), 4, col, None)
        d.text((798,y-1), str(j+1), font=F_SMALL, fill=MUTED)

    # Output result card
    d.text((859,184), 'TOP MATCH', font=F_SMALL, fill=GREEN)
    d.text((859,213), 'Backend', font=F_HEAD, fill=TEXT)
    d.text((859,233), 'вакансия', font=F_BODY, fill=TEXT)
    rounded(d, (859,277,936,297), 6, (18,47,42), None)
    centered(d,(897,287),'selected',F_SMALL,GREEN)

    # Traveling packets: looped, bright pulses on each connection.
    for start,end,col,phase in [(250,286,CYAN,0.0),(535,571,BLUE,0.29),(809,845,VIOLET,0.57)]:
        q=(t*2.2+phase)%1
        x=start+(end-start)*q
        r=3.2+1.2*(0.5+0.5*math.sin(2*math.pi*(t*2.2+phase)))
        d.ellipse((x-r-5,240-r-5,x+r+5,240+r+5), fill=tuple(int(v*.23) for v in col))
        d.ellipse((x-r,240-r,x+r,240+r), fill=col)
    # Animated vertical scan/highlight traverses shortlist -> semantic rank; seamless sinusoidal pulse.
    pulse=(math.sin(2*math.pi*t)+1)/2
    d.line((304, 233, 517, 233), fill=(32+int(19*pulse),48+int(22*pulse),68+int(27*pulse)), width=1)
    # cycle indicator / accurate model note
    d.line((34, 373, 966, 373), fill=EDGE, width=1)
    d.text((34, 391), 'REQUEST', font=F_SMALL, fill=MUTED)
    d.text((107, 390), 'резюме + вакансии', font=F_BODY, fill=TEXT)
    d.text((379, 391), 'RULES → BM25 → EMBEDDING RERANK', font=F_SMALL, fill=(172,190,210))
    d.text((966, 391), 'No scores shown', font=F_SMALL, fill=MUTED, anchor='ra')
    # moving progress dash on lower rail
    x=34+(932*((t*1.0)%1))
    d.line((34, 424, 966, 424), fill=(24,37,53), width=2)
    d.line((x, 424, min(966,x+48), 424), fill=CYAN, width=2)
    d.text((34, 440), 'VRSCORE  ·  СХЕМА ПОИСКА ВАКАНСИЙ', font=F_SMALL, fill=(91,112,139))
    return im


def main():
    frames=[frame(i) for i in range(FRAMES)]
    out=HERE/'concept-2.gif'
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=170, loop=0, optimize=True, disposal=2)
    print(f'Wrote {out} ({W}x{H}, {FRAMES} frames, {FRAMES*170/1000:.2f}s loop)')

if __name__=='__main__': main()
