"""Public Step-1 interface: parse, execute, or run(question, context)."""
from .context_parser import parse_context
from .query_parser import parse
from .executor import execute
from .schema import ExecutionResult, QueryError, StructuredQuery


def run(question: str, context: str) -> ExecutionResult:
    parsed = parse_context(context)
    try:
        query = parse(question, parsed)
    except QueryError as error:
        return ExecutionResult(error.status, None, reason=error.reason, stage=error.stage,
                               trace={'execution_status': error.status, 'final_answer': None})
    return execute(query, parsed)


__all__ = ['parse', 'parse_context', 'execute', 'run', 'StructuredQuery', 'ExecutionResult']
