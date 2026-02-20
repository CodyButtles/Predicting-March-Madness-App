(function () {
  const payload = window.MM_PAYLOAD;
  const root = document.getElementById('mm-root');
  if (!payload || !root) return;

  // Ask Streamlit to resize this iframe to better use vertical space.
  // This helps avoid large blank space below when the browser zoom changes.
  function requestFrameHeight(px) {
    const height = Math.max(420, Math.round(Number(px) || 0));
    try {
      window.parent.postMessage(
        {
          isStreamlitMessage: true,
          type: 'streamlit:setFrameHeight',
          height,
        },
        '*',
      );
    } catch (e) {
      // no-op (non-Streamlit context)
    }
  }

  function autoFrameHeight() {
    let parentH = null;
    try {
      parentH = window.parent && window.parent.innerHeight ? Number(window.parent.innerHeight) : null;
    } catch (e) {
      parentH = null;
    }
    const base = Number.isFinite(parentH) && parentH > 0 ? parentH : window.innerHeight;
    const desired = Math.max(560, Math.min(1100, Math.round(base * 0.92)));
    requestFrameHeight(desired);
  }

  const adv = payload.adv || {};
  const matchup = payload.matchup || {};

  const ROUND_LABELS = [
    { key: 'R64', label: 'Round of 64' },
    { key: 'R32', label: 'Round of 32' },
    { key: 'S16', label: 'Sweet 16' },
    { key: 'E8', label: 'Elite 8' },
    { key: 'CH', label: 'Region Champ' },
  ];

  const REGION_NAME = {
    UL: 'Upper Left',
    UR: 'Upper Right',
    LL: 'Lower Left',
    LR: 'Lower Right',
  };

  function slugToName(slug) {
    const parts = String(slug || '').split('/').filter(Boolean);
    const raw = parts.length ? parts[parts.length - 1] : String(slug || '');
    return raw
      .split('-')
      .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : w))
      .join(' ');
  }

  function pct(x, digits = 0) {
    if (x === undefined || x === null || Number.isNaN(x)) return '—';
    return `${Number(x).toFixed(digits)}%`;
  }

  function getProb(teamA, teamB) {
    try {
      if (matchup[teamA] && matchup[teamA][teamB] !== undefined) return Number(matchup[teamA][teamB]);
      if (matchup[teamB] && matchup[teamB][teamA] !== undefined) return 1 - Number(matchup[teamB][teamA]);
    } catch (e) {}
    return 0.5;
  }

  function confidence(p) {
    const x = Math.abs(Number(p) - 0.5) * 2;
    return Math.max(0, Math.min(1, x));
  }

  const tip = document.createElement('div');
  tip.className = 'mm-tooltip';
  document.body.appendChild(tip);

  function renderTooltip(teamSlug, opponentSlug, pWin) {
    const a = adv[teamSlug] || {};
    const hasMatchup = opponentSlug && pWin !== null && pWin !== undefined && !Number.isNaN(pWin);
    const conf = hasMatchup ? confidence(pWin) : null;
    tip.innerHTML = `
      <div class="title">${slugToName(teamSlug)}</div>
      <table>
        ${hasMatchup ? `<tr><td>Win vs ${slugToName(opponentSlug)}</td><td>${pct(Number(pWin) * 100, 1)}</td></tr>` : ''}
        ${hasMatchup ? `<tr><td>Confidence (|p-0.5|)</td><td>${pct(Number(conf) * 100, 0)}</td></tr>` : ''}
        <tr><td>Reach R32</td><td>${pct(a.R32, 1)}</td></tr>
        <tr><td>Reach S16</td><td>${pct(a.S16, 1)}</td></tr>
        <tr><td>Reach E8</td><td>${pct(a.E8, 1)}</td></tr>
        <tr><td>Reach F4</td><td>${pct(a.F4, 1)}</td></tr>
        <tr><td>Reach Final</td><td>${pct(a.Final, 1)}</td></tr>
        <tr><td>Win Champ</td><td>${pct(a.Champ, 1)}</td></tr>
      </table>
    `;
  }

  function moveTip(e) {
    const pad = 14;
    tip.style.left = e.clientX + pad + 'px';
    tip.style.top = e.clientY + pad + 'px';
  }

  function showTip(e) {
    tip.style.display = 'block';
    moveTip(e);
  }

  function hideTip() {
    tip.style.display = 'none';
  }

  // Bracket model (per region): rounds[0]=8 matches, rounds[1]=4, rounds[2]=2, rounds[3]=1.
  function emptyRoundsFromRound1(round1) {
    const rounds = [];
    rounds.push(round1.map((m) => ({ teams: m.map((t) => ({ ...t })), winner: null })));
    rounds.push(new Array(4).fill(0).map(() => ({ teams: [null, null], winner: null })));
    rounds.push(new Array(2).fill(0).map(() => ({ teams: [null, null], winner: null })));
    rounds.push(new Array(1).fill(0).map(() => ({ teams: [null, null], winner: null })));
    return rounds;
  }

  function setWinner(regionState, roundIndex, matchIndex, slotIndex) {
    const match = regionState.rounds[roundIndex][matchIndex];
    const team = match.teams[slotIndex];
    if (!team) return;

    match.winner = team.team;

    // Propagate winner to next round slot
    if (roundIndex < 3) {
      const nextRound = roundIndex + 1;
      const nextMatchIndex = Math.floor(matchIndex / 2);
      const nextSlotIndex = matchIndex % 2;

      const nextMatch = regionState.rounds[nextRound][nextMatchIndex];
      nextMatch.teams[nextSlotIndex] = { ...team };
      nextMatch.winner = null;

      // Clear downstream from this next match
      clearDownstream(regionState, nextRound, nextMatchIndex);
    }
  }

  function clearDownstream(regionState, fromRound, fromMatch) {
    for (let r = fromRound; r < regionState.rounds.length; r++) {
      const matches = regionState.rounds[r];
      for (let m = 0; m < matches.length; m++) {
        if (r === fromRound && m !== fromMatch) continue;
        const match = matches[m];
        if (r > fromRound) {
          match.teams = [null, null];
        }
        match.winner = null;
      }
      fromMatch = Math.floor(fromMatch / 2);
    }
  }

  function renderMatch(match, onPick, roundIndex) {
    const el = document.createElement('div');
    el.className = 'mm-match';

    if (roundIndex === 0) {
      const isPending = !!(match && match.teams && match.teams[0] && match.teams[1] && !match.winner);
      if (isPending) el.classList.add('mm-r1-pending');
      else if (match && match.winner) el.classList.add('mm-r1-done');
    }

    const a = match.teams[0];
    const b = match.teams[1];

    let pA = null;
    let pB = null;
    if (a && b) {
      pA = getProb(a.team, b.team);
      pB = 1 - pA;
    }

    [a, b].forEach((teamObj, idx) => {
      const row = document.createElement('div');
      row.className = 'mm-team';

      if (teamObj && match.winner === teamObj.team) row.classList.add('mm-picked');

      row.innerHTML = `
        <div class="mm-seed">${teamObj ? teamObj.seed : ''}</div>
        <div class="mm-name">${teamObj ? slugToName(teamObj.team) : '—'}</div>
        <div class="mm-metrics">${teamObj && a && b ? `${pct((idx === 0 ? pA : pB) * 100, 0)} | c${pct(confidence(idx === 0 ? pA : pB) * 100, 0)}` : ''}</div>
      `;

      if (teamObj) {
        row.addEventListener('mouseenter', (e) => {
          const opponent = idx === 0 ? (b ? b.team : null) : (a ? a.team : null);
          const pWin = idx === 0 ? pA : pB;
          renderTooltip(teamObj.team, opponent, pWin);
          showTip(e);
        });
        row.addEventListener('mousemove', moveTip);
        row.addEventListener('mouseleave', hideTip);

        row.addEventListener('click', () => onPick(idx));
      }

      el.appendChild(row);
    });

    return el;
  }

  function renderRegion(regionState) {
    const regionEl = document.createElement('div');
    regionEl.className = 'mm-region';
    regionEl.classList.add(`mm-region--${regionState.key}`);

    const isRight = regionState.key === 'UR' || regionState.key === 'LR';
    if (isRight) regionEl.classList.add('mm-region--right');

    // Intentionally omit the region header (e.g., "Lower Right") to keep the layout cleaner.

    const bracketEl = document.createElement('div');
    bracketEl.className = 'mm-bracket';
    regionEl.appendChild(bracketEl);

    function rerender() {
      bracketEl.innerHTML = '';

      function buildRoundCol(r) {
        const col = document.createElement('div');
        col.className = 'mm-col';
        col.dataset.round = ROUND_LABELS[r].key;
        col.classList.add(`mm-round-${ROUND_LABELS[r].key}`);

        const t = document.createElement('div');
        t.className = 'mm-col-title';
        t.textContent = ROUND_LABELS[r].label;
        col.appendChild(t);

        regionState.rounds[r].forEach((match, matchIndex) => {
          const mEl = renderMatch(
            match,
            (slotIndex) => {
              setWinner(regionState, r, matchIndex, slotIndex);
              rerender();
              renderFinalFour();
            },
            r,
          );
          col.appendChild(mEl);
        });

        return col;
      }

      function buildChampCol() {
        const champCol = document.createElement('div');
        champCol.className = 'mm-col';
        champCol.dataset.round = 'CH';
        champCol.classList.add('mm-round-CH');

        const champ = regionState.rounds[3][0].winner;
        const champBox = document.createElement('div');
        champBox.className = 'mm-match mm-champ-box';
        champBox.innerHTML = `<div class="mm-team mm-picked"><div class="mm-seed"></div><div class="mm-name">${champ ? slugToName(champ) : '—'}</div><div class="mm-metrics"></div></div>`;
        champCol.appendChild(champBox);

        return champCol;
      }

      if (!isRight) {
        // Left side: R64 → ... → Region Champ
        for (let r = 0; r < 4; r++) bracketEl.appendChild(buildRoundCol(r));
        bracketEl.appendChild(buildChampCol());
      } else {
        // Right side: mirror so the bracket flows toward the center (ESPN-style)
        bracketEl.appendChild(buildChampCol());
        for (let r = 3; r >= 0; r--) bracketEl.appendChild(buildRoundCol(r));
      }

      // Make later rounds form a sideways pyramid by distributing matches vertically.
      // We size all columns to the Round-of-64 column height so justify-content can work.
      window.requestAnimationFrame(() => {
        const r64Col = bracketEl.querySelector('.mm-round-R64');
        if (!r64Col) return;
        const h = Math.ceil(r64Col.getBoundingClientRect().height);
        if (!Number.isFinite(h) || h <= 0) return;
        bracketEl.style.setProperty('--mm-col-h', `${h}px`);
      });
    }

    rerender();
    return regionEl;
  }

  // Build state
  const regionStates = (payload.regions || []).map((r) => ({
    key: r.key,
    rounds: emptyRoundsFromRound1(r.round1 || []),
  }));

  // Header + regions
  const header = document.createElement('div');
  header.className = 'mm-header';
  header.innerHTML = `<h2>Interactive Bracket Builder</h2><div class="meta">Click a team to advance. Hover for round odds.</div>`;
  root.appendChild(header);

  // Single scroll canvas (avoid each region feeling like its own scroll box)
  const canvas = document.createElement('div');
  canvas.className = 'mm-canvas';
  root.appendChild(canvas);

  const grid = document.createElement('div');
  grid.className = 'mm-grid';
  canvas.appendChild(grid);

  // Internal zoom (avoid browser zoom causing odd whitespace). Use Ctrl+wheel over the bracket.
  const ZOOM_KEY = 'mm_bracket_zoom_v1';
  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  function getZoom() {
    const raw = Number(localStorage.getItem(ZOOM_KEY));
    return Number.isFinite(raw) ? raw : 0.9;
  }

  function setZoom(z) {
    const next = clamp(Number(z), 0.6, 1.15);
    grid.style.zoom = String(next);
    localStorage.setItem(ZOOM_KEY, String(next));
    autoFrameHeight();
  }

  setZoom(getZoom());

  canvas.addEventListener(
    'wheel',
    (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const current = getZoom();
      const delta = e.deltaY > 0 ? -0.05 : 0.05;
      setZoom(current + delta);
    },
    { passive: false },
  );

  // Initial + responsive sizing
  autoFrameHeight();
  window.addEventListener('resize', () => {
    // Throttle via rAF to avoid spamming postMessage during resize
    window.requestAnimationFrame(autoFrameHeight);
  });

  function appendRegion(regionKey) {
    const rs = regionStates.find((x) => x.key === regionKey);
    if (rs) grid.appendChild(renderRegion(rs));
  }

  // ESPN-style: UL/LL on left, UR/LR on right, Final Four in the center
  appendRegion('UL');
  appendRegion('UR');
  appendRegion('LL');
  appendRegion('LR');

  // Final Four / Champion (built from region champions)
  const ff = document.createElement('div');
  ff.className = 'mm-finalfour';
  ff.innerHTML = `<h3>Final Four</h3><div class="mm-ff-grid" id="mm-ff-grid"></div>`;
  grid.appendChild(ff);

  const ffGrid = ff.querySelector('#mm-ff-grid');
  const ffState = {
    semi1: { teams: [null, null], winner: null },
    semi2: { teams: [null, null], winner: null },
    champ: { teams: [null, null], winner: null },
  };

  function sameTeam(a, b) {
    const ta = a && a.team ? a.team : null;
    const tb = b && b.team ? b.team : null;
    return ta === tb;
  }

  function setTeamsIfChanged(matchState, newTeams) {
    const oldTeams = matchState.teams || [null, null];
    const changed = !sameTeam(oldTeams[0], newTeams[0]) || !sameTeam(oldTeams[1], newTeams[1]);
    matchState.teams = newTeams;
    if (changed) matchState.winner = null;
    return changed;
  }

  function regionChamp(regionKey) {
    const r = regionStates.find((x) => x.key === regionKey);
    return r && r.rounds[3] && r.rounds[3][0] ? r.rounds[3][0].winner : null;
  }

  function renderFinalFour() {
    ffGrid.innerHTML = '';

    const ul = regionChamp('UL');
    const ur = regionChamp('UR');
    const ll = regionChamp('LL');
    const lr = regionChamp('LR');

    // Quadrants are layout positions:
    // - UL/LL are the left side of the printed bracket
    // - UR/LR are the right side of the printed bracket
    // The national semifinals are therefore left-vs-left and right-vs-right.
    const semi1Changed = setTeamsIfChanged(ffState.semi1, [ul ? { seed: '', team: ul } : null, ll ? { seed: '', team: ll } : null]);
    const semi2Changed = setTeamsIfChanged(ffState.semi2, [ur ? { seed: '', team: ur } : null, lr ? { seed: '', team: lr } : null]);

    // If semifinal participants changed, clear anything downstream.
    if (semi1Changed || semi2Changed) {
      ffState.champ.teams = [null, null];
      ffState.champ.winner = null;
    }

    // If a previous winner is no longer a valid participant, clear it.
    if (ffState.semi1.winner && ![ul, ll].includes(ffState.semi1.winner)) ffState.semi1.winner = null;
    if (ffState.semi2.winner && ![ur, lr].includes(ffState.semi2.winner)) ffState.semi2.winner = null;

    // Champion matchup is derived from semifinal winners.
    const champ0 = ffState.semi1.winner ? ffState.semi1.teams.find((t) => t && t.team === ffState.semi1.winner) : null;
    const champ1 = ffState.semi2.winner ? ffState.semi2.teams.find((t) => t && t.team === ffState.semi2.winner) : null;
    const champChanged = setTeamsIfChanged(ffState.champ, [champ0 ? { ...champ0 } : null, champ1 ? { ...champ1 } : null]);
    if (champChanged) ffState.champ.winner = null;

    const semi1El = renderMatch(ffState.semi1, (slotIdx) => {
      const t = ffState.semi1.teams[slotIdx];
      if (!t) return;
      ffState.semi1.winner = t.team;
      renderFinalFour();
    }, null);
    semi1El.classList.add('mm-ff-semi');

    const semi2El = renderMatch(ffState.semi2, (slotIdx) => {
      const t = ffState.semi2.teams[slotIdx];
      if (!t) return;
      ffState.semi2.winner = t.team;
      renderFinalFour();
    }, null);
    semi2El.classList.add('mm-ff-semi');

    const champEl = renderMatch(ffState.champ, (slotIdx) => {
      const t = ffState.champ.teams[slotIdx];
      if (!t) return;
      ffState.champ.winner = t.team;
      renderFinalFour();
    }, null);
    champEl.classList.add('mm-ff-champ');

    const champWrap = document.createElement('div');
    champWrap.className = 'mm-ff-champwrap';
    champWrap.innerHTML = `<div class="mm-col-title">Championship</div>`;
    champWrap.appendChild(champEl);

    const champBox = document.createElement('div');
    champBox.className = 'mm-match mm-ff-winner';
    champBox.innerHTML = `<div class="mm-col-title">Champion</div><div style="margin-top:6px; font-weight:700;">${ffState.champ.winner ? slugToName(ffState.champ.winner) : '—'}</div>`;

    // Layout: semis side-by-side, then championship, then champion display.
    ffGrid.appendChild(semi1El);
    ffGrid.appendChild(semi2El);
    ffGrid.appendChild(champWrap);
    ffGrid.appendChild(champBox);
  }

  renderFinalFour();
})();
