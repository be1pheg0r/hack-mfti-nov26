"""Browser-level checks for the demo's responsive document and stage layout."""

import html as html_module
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "vrscore" / "web" / "demo"


def test_long_documents_and_rankings_stay_inside_responsive_layout(tmp_path):
    candidates = sorted(
        (Path.home() / ".cache/ms-playwright").glob(
            "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
        )
    )
    chrome = shutil.which("chromium") or (str(candidates[-1]) if candidates else None)
    if chrome is None:
        pytest.skip("Chromium is not installed")
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"<link[^>]*fonts\.googleapis\.com[^>]*>", "", html)
    html = html.replace('href="/fsm.css"', f'href="{(ASSETS / "fsm.css").as_uri()}"')
    css = Path(os.environ.get("VRSCORE_TEST_STYLES", ASSETS / "styles.css"))
    html = html.replace('href="/styles.css"', f'href="{css.as_uri()}"')
    html = html.replace('<script src="/journey.js" defer></script>', "")
    html = html.replace(
        "</body>",
        """<script>
        document.querySelector('#login').hidden=true;
        document.querySelector('#app').hidden=false;
        document.querySelector('#preview-name').textContent='ДлинноеИмя'.repeat(20);
        document.querySelector('#preview-title').textContent='Должность'.repeat(30);
        document.querySelector('#preview-text').textContent=
          'Длинное резюме '.repeat(600);
        document.querySelector('#resume-list').innerHTML=Array.from({length:10},
          (_,i)=>'<button class="resume-choice"><span>'+i+'</span><span><strong>'+
          'Имя'.repeat(80)+'</strong></span><b>↗</b></button>').join('');
        const rect=s=>document.querySelector(s).getBoundingClientRect();
        const doc=document.querySelector('.document-body');
        const choice=document.querySelector('.resume-choice');
        const choose={panel:rect('.document').bottom<=rect('#choose').bottom+1,
          button:rect('#analyze').bottom<=rect('#choose').bottom+1,
          scroll:doc.scrollHeight>doc.clientHeight,
          docFits:doc.scrollWidth<=doc.clientWidth,
          choiceFits:choice.scrollWidth<=choice.clientWidth};
        document.querySelector('#choose').hidden=true;
        document.querySelector('#journey').hidden=false;
        document.querySelector('#stage-description').textContent='Описание этапа';
        document.querySelector('#stages').innerHTML=Array.from({length:4},
          (_,i)=>'<li><span>'+i+'</span><strong>ЭТАП РАНЖИРОВАНИЯ</strong></li>'
        ).join('');
        document.querySelector('#jobs').innerHTML=Array.from({length:27},
          (_,i)=>'<div class="job-row"><span>'+i+'</span><span class="job-name">'+
          'ДлиннаяВакансия'.repeat(20)+'</span><span>0.123</span></div>').join('');
        const stage={panel:rect('.leaderboard').bottom<=rect('#journey').bottom+1,
          button:rect('#next').bottom<=rect('.stage-panel').bottom+1,
          beforeSlide:rect('#journey').bottom<=rect('#explain').top+1,
          horizontal:document.documentElement.scrollWidth<=innerWidth};
        document.querySelector('.scroll-cue').click();
        stage.slideOpens=rect('#explain').top<innerHeight;
        document.body.setAttribute('data-checks',JSON.stringify({choose,stage}));
        </script></body>""",
    )
    page = tmp_path / "layout.html"
    page.write_text(html, encoding="utf-8")
    for width, height in ((1366, 768), (1024, 600), (850, 600), (390, 844)):
        result = subprocess.run(
            [
                chrome,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--allow-file-access-from-files",
                f"--window-size={width},{height}",
                "--dump-dom",
                page.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        match = re.search(r'data-checks="([^"]+)"', result.stdout)
        assert match, result.stderr
        checks = json.loads(html_module.unescape(match.group(1)))
        assert all(checks["choose"].values()), (width, height, checks)
        assert all(checks["stage"].values()), (width, height, checks)
