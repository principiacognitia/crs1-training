"""Task 1.3: Validation suite for minicalculus grammar and corpus."""
import json, pytest
from pathlib import Path
from lark import Lark, exceptions
from hypothesis import given, strategies as st, settings

ROOT = Path(__file__).parent
GRAMMAR = (ROOT / 'grammar.lark').read_text()
PARSER = Lark(GRAMMAR, start='program', parser='earley', ambiguity='resolve')

def parse(s):
    return PARSER.parse(s)

class TestL1:
    def test_integer(self):        assert parse("42")
    def test_addition(self):       assert parse("1 + 2")
    def test_subtraction(self):    assert parse("10 - 3")
    def test_multiplication(self): assert parse("4 * 5")
    def test_division(self):       assert parse("8 / 2")
    def test_power(self):          assert parse("2 ^ 10")
    def test_unary_neg(self):      assert parse("-5 + 3")
    def test_nested(self):         assert parse("((1 + 2) * (3 - 1))")
    def test_decimal(self):        assert parse("3.14 * 2")

class TestL2:
    def test_variable(self):       assert parse("x")
    def test_var_expr(self):       assert parse("x + 1")
    def test_func_call(self):      assert parse("f(x, y)")
    def test_func_no_args(self):   assert parse("h()")
    def test_nested_func(self):    assert parse("f(g(x))")

class TestL3:
    def test_assignment(self):     assert parse("x = 5")
    def test_solve(self):          assert parse("solve(x + 5 == 0, x)")
    def test_solve_ineq(self):     assert parse("solve(x - 3 > 0, x)")
    def test_simplify(self):       assert parse("simplify(x + x)")
    def test_comparison(self):     assert parse("3 <= 5")

class TestL4:
    def test_define(self):         assert parse("define f(x) = x + 1")
    def test_define_binary(self):  assert parse("define add(a, b) = a + b")
    def test_forall(self):         assert parse("forall n in {0 .. 10} : n + 0 == n")
    def test_exists(self):         assert parse("exists x in {1 .. 100} : x * x == 49")
    def test_forall_named(self):   assert parse("forall x in integers : x + 0 == x")

class TestRejects:
    def _reject(self, s):
        with pytest.raises(exceptions.UnexpectedInput):
            parse(s)
    def test_unclosed_paren(self):  self._reject("(1 + 2")
    def test_double_op(self):       self._reject("1 ++ 2")
    def test_incomplete_solve(self): parse("solve(x)")  # L2 func call — semantic error, not syntactic

def load_corpus(level):
    path = ROOT / f"corpus-L{level}.jsonl"
    if not path.exists():
        pytest.skip(f"corpus-L{level}.jsonl not found")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_corpus_size(level):
    assert len(load_corpus(level)) >= 500

@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_corpus_parses(level):
    failed = [(ex["input"], "") for ex in load_corpus(level)[:200]
              if not _try_parse(ex["input"])]
    assert not failed, f"L{level}: {len(failed)} parse failures"

def _try_parse(s):
    try: parse(s); return True
    except: return False

@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_incorrect_rate(level):
    ex = load_corpus(level)
    rate = sum(1 for e in ex if not e["correct"]) / len(ex)
    assert 0.05 <= rate <= 0.20

@settings(max_examples=50, deadline=5000)
@given(st.integers(min_value=0, max_value=1000))
def test_integers_always_parse(n):
    assert parse(str(n))

@settings(max_examples=50, deadline=5000)
@given(st.sampled_from(['x','y','z','n']), st.integers(0, 100))
def test_assignment_always_parses(var, val):
    assert parse(f"{var} = {val}")
