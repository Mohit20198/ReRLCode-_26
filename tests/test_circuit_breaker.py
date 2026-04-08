"""
tests/test_circuit_breaker.py
──────────────────────────────
Tests for the circuit breaker safety rail.
"""

import time
import pytest
from orchestrator.circuit_breaker import CircuitBreaker


def test_new_breaker_is_closed():
    cb = CircuitBreaker(max_migrations=3, window_minutes=10)
    assert not cb.is_open("job-001")


def test_opens_after_max_migrations():
    cb = CircuitBreaker(max_migrations=3, window_minutes=10)
    for _ in range(3):
        cb.record_migration("job-001")
    assert cb.is_open("job-001")


def test_does_not_open_before_max():
    cb = CircuitBreaker(max_migrations=3, window_minutes=10)
    cb.record_migration("job-001")
    cb.record_migration("job-001")
    assert not cb.is_open("job-001")


def test_force_open_blocks_agent():
    cb = CircuitBreaker(max_migrations=10, window_minutes=10)
    cb.force_open("job-002", duration_minutes=60)
    assert cb.is_open("job-002")


def test_force_reset_reopens():
    cb = CircuitBreaker(max_migrations=3, window_minutes=10)
    for _ in range(3):
        cb.record_migration("job-003")
    assert cb.is_open("job-003")
    cb.force_reset("job-003")
    assert not cb.is_open("job-003")


def test_different_jobs_are_independent():
    cb = CircuitBreaker(max_migrations=3, window_minutes=10)
    for _ in range(3):
        cb.record_migration("job-A")
    assert cb.is_open("job-A")
    assert not cb.is_open("job-B")


def test_get_status_returns_correct_state():
    cb = CircuitBreaker(max_migrations=3, window_minutes=10)
    cb.record_migration("job-X")
    status = cb.get_status("job-X")
    assert status["state"] == "CLOSED"
    assert status["migrations_in_window"] == 1
    assert status["max_migrations"] == 3
