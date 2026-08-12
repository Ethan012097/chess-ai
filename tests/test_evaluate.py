"""`evaluate.py` 的統計計算測試。

重點是 Elo 信賴區間。這裡曾經有個很難察覺的 bug：用課本上的 Wald 區間時，
得分率逼近 0 或 1 會讓變異數塌成 0，區間縮成一個點，**點估計反而跑到區間外面**
（M7 的 198勝2和0負 印出 `+920（CI: +768 ~ +800）`）。

改用 Wilson 區間之後，`lower <= point <= upper` 是恆成立的不變式。
"""

from __future__ import annotations

import math

import pytest

from src.evaluate import (
    ELO_CLAMP,
    MatchResult,
    elo_confidence_interval,
    elo_difference,
    format_elo,
    parse_cutechess_result,
    report_terminations,
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
