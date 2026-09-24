"""`evaluate.py` 的統計計算測試。

重點是 Elo 信賴區間。這裡曾經有個很難察覺的 bug：用課本上的 Wald 區間時，
得分率逼近 0 或 1 會讓變異數塌成 0，區間縮成一個點，**點估計反而跑到區間外面**
（M7 的 198勝2和0負 印出 `+920（CI: +768 ~ +800）`）。

改用 Wilson 區間之後，`lower <= point <= upper` 是恆成立的不變式。
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine
import pytest

from src.evaluate import (
    ELO_CLAMP,
    MatchResult,
    elo_confidence_interval,
    elo_difference,
    format_elo,
    parse_cutechess_result,
    print_absolute_elo,
    report_terminations,
    MAX_ENGINE_RESTARTS,
    mcnemar_exact_p,
    StockfishMover,
    stockfish_opponents,
    wilson_interval,
)


# --- MatchResult ------------------------------------------------------------


def test_score_counts_draw_as_half() -> None:
    """得分率 = (勝 + 0.5 * 和) / 總局數。"""
    assert MatchResult(wins=10, draws=4, losses=6).score == pytest.approx(0.6)
    assert MatchResult(wins=0, draws=10, losses=0).score == pytest.approx(0.5)
    assert MatchResult(0, 0, 0).score == 0.0


def test_win_rate_excludes_draws() -> None:
    """勝率不含和局，跟得分率是兩回事。"""
    r = MatchResult(wins=10, draws=10, losses=0)
    assert r.win_rate == pytest.approx(0.5)
    assert r.score == pytest.approx(0.75)


# --- Elo 換算 ---------------------------------------------------------------


def test_elo_difference_known_values() -> None:
    """幾個標準對照值。"""
    assert elo_difference(0.5) == pytest.approx(0.0, abs=1e-9)
    # 得分率 0.75 約等於 +191 Elo
    assert elo_difference(0.75) == pytest.approx(190.8, abs=1.0)
    # 對稱性
    assert elo_difference(0.25) == pytest.approx(-elo_difference(0.75))


def test_elo_difference_is_monotonic() -> None:
    """得分率越高，Elo 差越大。"""
    scores = [0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99]
    elos = [elo_difference(s) for s in scores]
    assert elos == sorted(elos)


def test_elo_difference_clamps_degenerate_scores() -> None:
    """0 與 1 會讓 log10 發散，必須夾住而不是回傳 inf 或炸掉。"""
    assert elo_difference(0.0) == -ELO_CLAMP
    assert elo_difference(1.0) == ELO_CLAMP
    assert math.isfinite(elo_difference(0.0))
    assert math.isfinite(elo_difference(1.0))


# --- Wilson 區間 ------------------------------------------------------------


def test_wilson_contains_point_estimate() -> None:
    """Wilson 區間必須包住得分率本身（各種樣本數與極端值）。"""
    for n in (1, 2, 10, 100, 1000):
        for score in (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0):
            low, high = wilson_interval(score, n)
            assert low <= score <= high, f"score={score} n={n} 區間 [{low}, {high}]"


def test_wilson_narrows_with_more_games() -> None:
    """局數越多，區間越窄。"""
    widths = []
    for n in (10, 50, 200, 1000):
        low, high = wilson_interval(0.6, n)
        widths.append(high - low)
    assert widths == sorted(widths, reverse=True)


def test_wilson_stays_in_unit_range() -> None:
    """區間不能跑出 [0, 1]。"""
    for score in (0.0, 0.5, 1.0):
        low, high = wilson_interval(score, 5)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_is_informative_at_extremes() -> None:
    """全勝時 Wald 的區間會塌成一個點，Wilson 不會。

    這就是換掉 Wald 的理由。
    """
    low, high = wilson_interval(1.0, 200)
    assert high == pytest.approx(1.0)
    assert low < 1.0, "全勝的區間下界必須小於 1，否則是假的精確"
    assert low > 0.9, "200 局全勝，下界不該低到 0.9 以下"


def test_wilson_with_zero_games() -> None:
    """沒有對局時回傳最大不確定性，不能除以零。"""
    assert wilson_interval(0.0, 0) == (0.0, 1.0)


# --- 不變式：這是 P8 的驗收條件 ---------------------------------------------


@pytest.mark.parametrize(
    "wins,draws,losses",
    [
        (198, 2, 0),      # M7 實測，就是這組讓舊算法露餡
        (18, 3, 9),       # M9 實測
        (200, 0, 0),      # 全勝
        (0, 0, 200),      # 全敗
        (0, 200, 0),      # 全和（Wald 會給出假的零寬度區間）
        (100, 0, 100),    # 五五波
        (1, 0, 0),        # 極小樣本
        (2, 0, 0),
        (0, 0, 0),        # 沒有對局
        (50, 25, 25),
    ],
)
def test_elo_point_estimate_lies_inside_interval(
    wins: int, draws: int, losses: int
) -> None:
    """`lower <= point <= upper` 必須恆成立（規格 §3.2）。"""
    elo, low, high = elo_confidence_interval(MatchResult(wins, draws, losses))
    assert low <= elo <= high, (
        f"{wins}勝{draws}和{losses}負 → 點估計 {elo:.1f} "
        f"落在區間 [{low:.1f}, {high:.1f}] 外面"
    )


def test_all_draws_gives_nonzero_width_interval() -> None:
    """200 局全和：得分率剛好 0.5，但區間不該是零寬度。

    舊的 Wald 算法在這裡會算出變異數 0 → 區間 (+0, +0)，
    看起來像是「非常確定兩邊實力完全相同」，那是假的精確。
    """
    elo, low, high = elo_confidence_interval(MatchResult(0, 200, 0))
    assert elo == pytest.approx(0.0, abs=1e-6)
    assert high - low > 10.0, "全和的區間寬度不該接近 0"


def test_small_sample_interval_spans_zero() -> None:
    """2 局全勝在統計上什麼都證明不了，區間下界該落在 0 以下。"""
    _, low, high = elo_confidence_interval(MatchResult(2, 0, 0))
    assert low < 0 < high, "小樣本的區間應該要涵蓋『其實沒比較強』的可能"


def test_more_games_narrows_elo_interval() -> None:
    """同樣的得分率，局數越多區間越窄。"""
    narrow = elo_confidence_interval(MatchResult(600, 0, 400))
    wide = elo_confidence_interval(MatchResult(60, 0, 40))
    assert (narrow[2] - narrow[1]) < (wide[2] - wide[1])


# --- 顯示格式 ---------------------------------------------------------------


def test_format_elo_marks_clamped_values() -> None:
    """碰到上下限要用 ≥ / ≤ 標示，不要假裝那是點估計。"""
    assert format_elo(ELO_CLAMP) == f"≥ +{ELO_CLAMP:.0f}"
    assert format_elo(-ELO_CLAMP) == f"≤ -{ELO_CLAMP:.0f}"
    assert format_elo(150.4) == "+150"
    assert format_elo(-150.4) == "-150"


def test_format_elo_normalises_negative_zero() -> None:
    """-0.0 要印成 +0，不要印成難看的 -0。"""
    assert format_elo(-0.0) == "+0"
    assert format_elo(0.0) == "+0"


# --- 結束原因統計 -----------------------------------------------------------
#
# 為什麼要有這幾條：曾經有一場 80 局的引擎對打，其中 28 局是我們的引擎回了
# `bestmove 0000` 被判非法著法而輸掉的。比分是 39-39 看起來完全正常，
# 只有結束原因才看得出三分之一的局數根本沒在下棋。


CUTECHESS_LINES = [
    "Started game 1 of 4 (A vs B)",
    "Finished game 1 (A vs B): 1-0 {White mates}",
    "Finished game 2 (B vs A): 1/2-1/2 {Draw by 3-fold repetition}",
    "Finished game 3 (A vs B): 0-1 {White makes an illegal move: 0000}",
    "Finished game 4 (B vs A): 1-0 {White mates}",
    "Score of A vs B: 2 - 1 - 1  [0.625] 4",
    "Elo difference: 88.7 +/- 120.5, LOS: 92.0 %, DrawRatio: 25.0 %",
    "SPRT: llr 0.5 (17.0%), lbound -2.94, ubound 2.94",
    "Finished match",
]


def test_parse_counts_termination_reasons() -> None:
    """每局的結束原因要被統計出來。"""
    parsed = parse_cutechess_result(CUTECHESS_LINES)
    assert parsed["terminations"] == {
        "White mates": 2,
        "Draw by 3-fold repetition": 1,
        "White makes an illegal move: 0000": 1,
    }
    # 原本的欄位不能因此壞掉
    assert (parsed["wins"], parsed["losses"], parsed["draws"]) == (2, 1, 1)
    assert parsed["elo"] == pytest.approx(88.7)


def test_parse_without_finished_game_lines() -> None:
    """沒有逐局紀錄時不要多出一個空的 terminations 欄位。"""
    parsed = parse_cutechess_result(["Score of A vs B: 1 - 0 - 0  [1.000] 1"])
    assert "terminations" not in parsed


def test_report_warns_about_illegal_moves(capsys) -> None:
    """出現非法著法棄權時要大聲警告，並指出這是 bug 不是棋力問題。"""
    report_terminations(parse_cutechess_result(CUTECHESS_LINES))
    out = capsys.readouterr().out
    assert "警告" in out
    assert "非法著法" in out
    assert "test_uci.py" in out, "警告要告訴使用者下一步跑什麼"


def test_report_is_quiet_when_all_games_are_clean(capsys) -> None:
    """全部正常結束時只列分佈，不要有警告嚇人。"""
    clean = [line for line in CUTECHESS_LINES if "illegal" not in line]
    report_terminations(parse_cutechess_result(clean))
    out = capsys.readouterr().out
    assert "結束原因" in out
    assert "警告" not in out


def test_report_does_nothing_without_data(capsys) -> None:
    """沒有結束原因資料時完全不印，不要輸出空的區塊。"""
    report_terminations({})
    assert capsys.readouterr().out == ""


# --- UCI_Elo 標尺 -----------------------------------------------------------
#
# Skill Level 不是校準過的 Elo 刻度（它靠隨機挑著法變弱），
# UCI_LimitStrength + UCI_Elo 才是。這幾條守住兩者不會被搞混。


def test_skill_level_opponents_have_no_known_elo() -> None:
    """Skill Level 沒有校準過的 Elo 對應值，第三個欄位必須是 None。

    填一個猜的數字進去，絕對 Elo 推估就會憑空多出假的精確度。
    """
    opponents = stockfish_opponents([0, 3], None)
    assert [o[0] for o in opponents] == ["Skill Level 0", "Skill Level 3"]
    assert all(o[2] is None for o in opponents)
    # LimitStrength 要明確關掉，否則引擎預設開著會蓋掉 Skill Level
    assert all(o[1]["UCI_LimitStrength"] is False for o in opponents)


def test_uci_elo_opponents_carry_their_anchor() -> None:
    """UCI_Elo 對手要記住自己值多少 Elo，那是絕對推估的基準。"""
    opponents = stockfish_opponents(None, [1400, 1800])
    assert [o[2] for o in opponents] == [1400, 1800]
    for label, options, elo in opponents:
        assert options["UCI_LimitStrength"] is True
        assert options["UCI_Elo"] == elo
        # LimitStrength 要排在 Elo 前面送出去
        assert list(options) == ["UCI_LimitStrength", "UCI_Elo"]


def test_both_kinds_can_coexist() -> None:
    """兩種可以同時組出來（雖然 CLI 預設不混用）。"""
    opponents = stockfish_opponents([0], [1500])
    assert len(opponents) == 2
    assert opponents[0][2] is None and opponents[1][2] == 1500


def test_empty_input_gives_empty_list() -> None:
    assert stockfish_opponents(None, None) == []
    assert stockfish_opponents([], []) == []


def test_absolute_elo_shifts_by_anchor(capsys) -> None:
    """我方 Elo = 對手 Elo + Elo差。得分率 50 % 時就等於對手的 Elo。"""
    reports = [
        {"opponent_elo": 1600, "score": 0.5, "elo_diff": 0.0,
         "elo_ci_low": -50.0, "elo_ci_high": 50.0},
    ]
    print_absolute_elo(reports)
    out = capsys.readouterr().out
    assert "1600" in out
    assert "[1550, 1650]" in out, "信賴區間要跟著平移到絕對刻度上"


def test_absolute_elo_flags_inconsistent_estimates(capsys) -> None:
    """幾個估計值差很多時要講出來，不要假裝可以平均成一個數字。"""
    reports = [
        {"opponent_elo": 1400, "score": 0.9, "elo_diff": 382.0,
         "elo_ci_low": 300.0, "elo_ci_high": 470.0},
        {"opponent_elo": 2000, "score": 0.1, "elo_diff": -382.0,
         "elo_ci_low": -470.0, "elo_ci_high": -300.0},
    ]
    print_absolute_elo(reports)
    out = capsys.readouterr().out
    assert "1782" in out and "1618" in out
    assert "差距偏大" in out


def test_absolute_elo_ignores_unanchored_reports(capsys) -> None:
    """只有 Skill Level 的結果不能拿來推絕對 Elo，整段要跳過。"""
    print_absolute_elo([{"opponent_elo": None, "score": 0.65, "elo_diff": 108.0,
                         "elo_ci_low": -20.0, "elo_ci_high": 235.0}])
    assert capsys.readouterr().out == ""


# --- 引擎卡住的容錯 ---------------------------------------------------------
#
# 實測跑 450 局的評估時，Stockfish 有一步卡了四分多鐘不回應，
# python-chess 丟出 TimeoutError，整批測量就這樣沒了。
# 一次 hiccup 不該讓幾十分鐘的結果歸零。


def _mover_with(total_failures: int, exc: type[Exception] = TimeoutError):
    """建一個 StockfishMover，但把開引擎換成假的。

    `total_failures` 是**跨引擎累計**的失敗次數：重開之後仍然算在同一個額度裡。
    這樣 total_failures 很大時就能模擬「怎麼重開都救不回來」。
    """
    remaining = {"n": total_failures}
    engines: list[SimpleNamespace] = []

    def make_engine():
        state = SimpleNamespace(quit_calls=0)

        def play(board, limit):  # noqa: ANN001
            if remaining["n"] > 0:
                remaining["n"] -= 1
                raise exc("模擬引擎卡住")
            return SimpleNamespace(move=next(iter(board.legal_moves)))

        def quit_():
            state.quit_calls += 1

        state.play = play
        state.quit = quit_
        engines.append(state)
        return state

    mover = StockfishMover.__new__(StockfishMover)
    mover.engine_path = Path("fake/stockfish.exe")
    mover.options = {}
    mover.limit = None
    mover.restarts = 0
    mover._open = lambda: make_engine()
    mover.engine = mover._open()
    return mover, engines


def test_mover_returns_move_when_engine_is_healthy() -> None:
    """正常情況不該有任何重開。"""
    mover, _ = _mover_with(total_failures=0)
    assert mover.play(chess.Board()) in chess.Board().legal_moves
    assert mover.restarts == 0


def test_mover_restarts_after_timeout() -> None:
    """引擎卡住要重開再試，而不是讓整批評估掛掉。"""
    mover, engines = _mover_with(total_failures=1)
    move = mover.play(chess.Board())
    assert move in chess.Board().legal_moves
    assert mover.restarts == 1
    assert len(engines) == 2, "應該開了第二個引擎行程"
    assert engines[0].quit_calls == 1, "舊的引擎要被收掉，不能留殭屍行程"


def test_mover_gives_up_with_actionable_message() -> None:
    """一直失敗就要放棄，但錯誤訊息要說明下一步做什麼。"""
    mover, _ = _mover_with(total_failures=99)
    with pytest.raises(RuntimeError) as exc:
        mover.play(chess.Board())
    text = str(exc.value)
    assert "engine_movetime_ms" in text, "要告訴使用者可以調哪個設定"
    assert "rnbqkbnr" in text, "要附上出問題的盤面，才能重現"
    assert mover.restarts == MAX_ENGINE_RESTARTS


def test_mover_also_handles_engine_terminated() -> None:
    """引擎行程直接死掉也要能救回來，不是只處理 timeout。"""
    mover, _ = _mover_with(total_failures=1, exc=chess.engine.EngineTerminatedError)
    assert mover.play(chess.Board()) in chess.Board().legal_moves
    assert mover.restarts == 1


# --- McNemar 配對檢定 -------------------------------------------------------
#
# 為什麼需要它：兩個搜尋器跑的是同一批題目，那是配對資料。
# 用獨立樣本的標準誤（每桶 100 題 → 約 ±5 %）會嚴重高估雜訊，
# 把真的顯著的退步判成「落在雜訊範圍內」。


def test_mcnemar_perfect_agreement_is_not_significant() -> None:
    """完全沒有不一致對時 p = 1，不能因為樣本大就宣稱有差異。"""
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_symmetric_counts_give_p_one() -> None:
    """b = c 代表兩邊互有勝負且完全平衡，p 應該是 1。"""
    for n in (1, 5, 20):
        assert mcnemar_exact_p(n, n) == pytest.approx(1.0)


def test_mcnemar_matches_known_values() -> None:
    """對照手算的精確二項檢定值。"""
    # b=10, c=0 → p = 2 * C(10,0)/2^10 = 2/1024
    assert mcnemar_exact_p(10, 0) == pytest.approx(2 / 1024)
    # b=6, c=1 → p = 2 * (C(7,0)+C(7,1))/2^7 = 2*8/128
    assert mcnemar_exact_p(6, 1) == pytest.approx(2 * 8 / 128)
    # b=1, c=0 → p = 2 * 1/2 = 1.0（一對不一致什麼都證明不了）
    assert mcnemar_exact_p(1, 0) == pytest.approx(1.0)


def test_mcnemar_is_symmetric_in_arguments() -> None:
    """雙尾檢定，交換 b 與 c 結果相同。"""
    for b, c in [(3, 9), (0, 7), (12, 4)]:
        assert mcnemar_exact_p(b, c) == pytest.approx(mcnemar_exact_p(c, b))


def test_mcnemar_never_exceeds_one() -> None:
    """雙尾機率乘 2 之後可能超過 1，必須夾住。"""
    for b in range(0, 8):
        for c in range(0, 8):
            assert 0.0 <= mcnemar_exact_p(b, c) <= 1.0


def test_mcnemar_is_more_sensitive_than_independent_samples() -> None:
    """這條說明為什麼要換方法。

    情境：每桶 100 題，A 89 % 、B 82 %（README §6.7 的 <1000 桶）。
    獨立樣本的標準誤約 5 %，7 個百分點的差距會被判成雜訊。
    但如果那 7 個百分點全部來自不一致對（b=7, c=0），配對檢定會說它顯著。
    """
    p = mcnemar_exact_p(7, 0)
    assert p < 0.05, "b=7, c=0 應該是顯著的，獨立樣本檢定會漏掉"
    # 而如果兩邊互有勝負（b=11, c=4），同樣的淨差距就不顯著了
    assert mcnemar_exact_p(11, 4) > 0.05
