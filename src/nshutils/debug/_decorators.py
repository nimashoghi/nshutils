"""Define public decorators."""

from __future__ import annotations

import functools
import inspect
import reprlib
import traceback
from collections.abc import Callable
from typing import Any, TypeAlias, cast

from . import _checkers
from ._config import ROOT_PATH, enabled
from ._globals import CallableT, ClassT, ExceptionT, aRepr
from ._types import Contract, Invariant, InvariantCheckEvent, Snapshot

EnabledInputType: TypeAlias = bool | Callable[[], bool]
ENABLED_DEFAULT = lambda: enabled(path=ROOT_PATH)


def _resolve_enabled(
    enabled_input: EnabledInputType | None = None,
    path_input: str | None = None,
) -> Callable[[], bool]:
    # If enabled is not set, we use `enabled(path)`.
    # If path is also not set, we use `enabled(ROOT_PATH)`.
    # If both are set, we should throw an error.
    match enabled_input, path_input:
        case None, None:
            return ENABLED_DEFAULT
        case None, _:
            return lambda: enabled(path=path_input)
        case _, None:
            return lambda: enabled(path=ROOT_PATH)
        case _:
            raise ValueError("You can't specify both 'enabled' and 'path'.")


def _wrap_with_enabled_check(
    func_orig: CallableT,
    func_wrapped: CallableT,
    enabled: Callable[[], bool],
):
    @functools.wraps(func_orig)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal func_wrapped, func_orig, enabled
        if enabled():
            return func_wrapped(*args, **kwargs)
        return func_orig(*args, **kwargs)

    return cast(CallableT, wrapped)


class require:  # pylint: disable=invalid-name
    """
    Decorate a function with a precondition.

    The arguments of the precondition are expected to be a subset of the arguments of the wrapped function.
    """

    def __init__(
        self,
        condition: Callable[..., Any],
        description: str | None = None,
        a_repr: reprlib.Repr = aRepr,
        enabled: EnabledInputType | None = None,
        path: str | None = None,
        error: Callable[..., ExceptionT]
        | type[ExceptionT]
        | BaseException
        | None = None,
    ) -> None:
        """
        Initialize.

        :param condition: precondition predicate

            If the condition returns a coroutine, you must specify the `error` as
            coroutines have side effects and can not be recomputed.
        :param description: textual description of the precondition
        :param a_repr: representation instance that defines how the values are represented
        :param enabled:
            The decorator is applied only if this argument is set.

            Otherwise, the condition check is disabled and there is no run-time overhead.

            The default is to always check the condition unless the interpreter runs in optimized mode (``-O`` or
            ``-OO``).
        :param error:
            The error is expected to denote either:

            * A callable. ``error`` is expected to accept a subset of function arguments and return an exception.
              The ``error`` is called on contract violation and the resulting exception is raised.
            * A subclass of ``BaseException`` which is instantiated with the violation message and raised
              on contract violation.
            * An instance of ``BaseException`` that will be raised with the traceback on contract violation.

        """
        enabled = _resolve_enabled(enabled, path)
        self.enabled = enabled
        self._contract = None  # type: Contract | None

        if error is None:
            pass
        elif isinstance(error, type):
            if not issubclass(error, BaseException):
                raise ValueError(
                    (
                        "The error of the contract is given as a type, "
                        "but the type does not inherit from BaseException: {}"
                    ).format(error)
                )
        else:
            if (
                not inspect.isfunction(error)
                and not inspect.ismethod(error)
                and not isinstance(error, BaseException)
            ):
                raise ValueError(
                    (
                        "The error of the contract must be either a callable (a function or a method), "
                        "a class (subclass of BaseException) or an instance of BaseException, but got: {}"
                    ).format(error)
                )

        location = None  # type: str | None
        tb_stack = traceback.extract_stack(limit=2)[:1]
        if len(tb_stack) > 0:
            frame = tb_stack[0]
            location = "File {}, line {} in {}".format(
                frame.filename, frame.lineno, frame.name
            )

        self._contract = Contract(
            condition=condition,
            description=description,
            a_repr=a_repr,
            error=error,
            location=location,
        )

    def __call__(self, func: CallableT) -> CallableT:
        """
        Add the precondition to the list of preconditions of the function ``func``.

        The function ``func`` is decorated with a contract checker if there is no contract checker in
        the decorator stack.

        :param func: function to be wrapped
        :return: contract checker around ``func`` if no contract checker on the decorator stack, or ``func`` otherwise
        """

        # Find a contract checker
        contract_checker = _checkers.find_checker(func=func)

        if contract_checker is None:
            # Wrap the function with a contract checker
            contract_checker = _checkers.decorate_with_checker(func=func)

        result = contract_checker

        assert self._contract is not None
        _checkers.add_precondition_to_checker(
            checker=contract_checker, contract=self._contract
        )

        return _wrap_with_enabled_check(func, result, self.enabled)


class snapshot:  # pylint: disable=invalid-name
    """
    Decorate a function with a snapshot of argument values captured *prior* to the function invocation.

    A snapshot is defined by a capture function (usually a lambda) that accepts one or more arguments of the function.
    If the name of the snapshot is not given, the capture function must have a single argument and the name is equal to
    the name of that single argument.

    The captured values are supplied to postconditions with the OLD argument of the condition and error function.
    Snapshots are inherited from the base classes and must not have conflicting names in the class hierarchy.
    """

    def __init__(
        self,
        capture: Callable[..., Any],
        name: str | None = None,
        enabled: EnabledInputType | None = None,
        path: str | None = None,
    ) -> None:
        """
        Initialize.

        :param capture:
            function to capture the snapshot accepting a one or more arguments of the original function *prior*
            to the execution
        :param name: name of the snapshot; if omitted, the name corresponds to the name of the input argument
        :param enabled:
            The decorator is applied only if ``enabled`` is set.

            Otherwise, the snapshot is disabled and there is no run-time overhead.

            The default is to always capture the snapshot unless the interpreter runs in optimized mode (``-O`` or
            ``-OO``).

        """
        enabled = _resolve_enabled(enabled, path)
        self._snapshot = None  # type: Snapshot | None
        self.enabled = enabled

        location = None  # type: str | None
        tb_stack = traceback.extract_stack(limit=2)[:1]
        if len(tb_stack) > 0:
            frame = tb_stack[0]
            location = "File {}, line {} in {}".format(
                frame.filename, frame.lineno, frame.name
            )

        self._snapshot = Snapshot(capture=capture, name=name, location=location)

    def __call__(self, func: CallableT) -> CallableT:
        """
        Add the snapshot to the list of snapshots of the function ``func``.

        The function ``func`` is expected to be decorated with at least one postcondition before the snapshot.

        :param func: function whose arguments we need to snapshot
        :return: ``func`` as given in the input
        """
        # Find a contract checker
        contract_checker = _checkers.find_checker(func=func)

        if contract_checker is None:
            raise ValueError(
                "You are decorating a function with a snapshot, but no postcondition was defined "
                "on the function before."
            )

        assert self._snapshot is not None, (
            "Expected the snapshot to have the property ``snapshot`` set."
        )

        _checkers.add_snapshot_to_checker(
            checker=contract_checker, snapshot=self._snapshot
        )

        return func


class ensure:  # pylint: disable=invalid-name
    """
    Decorate a function with a postcondition.

    The arguments of the postcondition are expected to be a subset of the arguments of the wrapped function.
    Additionally, the argument "result" is reserved for the result of the wrapped function. The wrapped function must
    not have "result" among its arguments.
    """

    def __init__(
        self,
        condition: Callable[..., Any],
        description: str | None = None,
        a_repr: reprlib.Repr = aRepr,
        enabled: EnabledInputType | None = None,
        path: str | None = None,
        error: (Callable[..., ExceptionT] | type[ExceptionT] | BaseException)
        | None = None,
    ) -> None:
        """
        Initialize.

        :param condition:
            postcondition predicate.

            If the condition returns a coroutine, you must specify the `error` as
            coroutines have side effects and can not be recomputed.
        :param description: textual description of the postcondition
        :param a_repr: representation instance that defines how the values are represented
        :param enabled:
            The decorator is applied only if this argument is set.

            Otherwise, the condition check is disabled and there is no run-time overhead.

            The default is to always check the condition unless the interpreter runs in optimized mode (``-O`` or
            ``-OO``).
        :param error:
            The error is expected to denote either:

            * A callable. ``error`` is expected to accept a subset of function arguments and return an exception.
              The ``error`` is called on contract violation and the resulting exception is raised.
            * A subclass of ``BaseException`` which is instantiated with the violation message and raised
              on contract violation.
            * An instance of ``BaseException`` that will be raised with the traceback on contract violation.
        """
        enabled = _resolve_enabled(enabled, path)
        self.enabled = enabled
        self._contract = None  # type: Contract | None

        if error is None:
            pass
        elif isinstance(error, type):
            if not issubclass(error, BaseException):
                raise ValueError(
                    (
                        "The error of the contract is given as a type, "
                        "but the type does not inherit from BaseException: {}"
                    ).format(error)
                )
        else:
            if (
                not inspect.isfunction(error)
                and not inspect.ismethod(error)
                and not isinstance(error, BaseException)
            ):
                raise ValueError(
                    (
                        "The error of the contract must be either a callable (a function or a method), "
                        "a class (subclass of BaseException) or an instance of BaseException, but got: {}"
                    ).format(error)
                )

        location = None  # type: str | None
        tb_stack = traceback.extract_stack(limit=2)[:1]
        if len(tb_stack) > 0:
            frame = tb_stack[0]
            location = "File {}, line {} in {}".format(
                frame.filename, frame.lineno, frame.name
            )

        self._contract = Contract(
            condition=condition,
            description=description,
            a_repr=a_repr,
            error=error,
            location=location,
        )

    def __call__(self, func: CallableT) -> CallableT:
        """
        Add the postcondition to the list of postconditions of the function ``func``.

        The function ``func`` is decorated with a contract checker if there is no contract checker in
        the decorator stack.

        :param func: function to be wrapped
        :return: contract checker around ``func`` if no contract checker on the decorator stack, or ``func`` otherwise
        """
        # Find a contract checker
        contract_checker = _checkers.find_checker(func=func)

        if contract_checker is None:
            # Wrap the function with a contract checker
            contract_checker = _checkers.decorate_with_checker(func=func)

        result = contract_checker

        assert self._contract is not None
        _checkers.add_postcondition_to_checker(
            checker=contract_checker, contract=self._contract
        )

        return _wrap_with_enabled_check(func, result, self.enabled)


class invariant:  # pylint: disable=invalid-name
    """
    Represent a class decorator to establish the invariant on all the public methods.

    Class method as well as "private" (prefix ``__``) and "protected" methods (prefix ``_``) may violate the invariant.
    Note that all magic methods (prefix ``__`` and suffix ``__``) are considered public and hence also need to establish
    the invariant. To avoid endless loops when generating the error message on an invariant breach, the method
    ``__repr__`` is deliberately exempt from observing the invariant.

    The invariant is checked *before* and *after* the method invocation.

    As invariants need to wrap dunder methods, including ``__init__``, their conditions *can not* be
    async, as most dunder methods need to be synchronous methods, and wrapping them with async code would
    break that constraint.

    For efficiency reasons, we do not check the invariants at the calls to ``__setattr__`` and ``__getattr__``
    methods by default. If you indeed want to check the invariants in those calls as well, make sure to adapt
    the argument ``check_on`` accordingly.
    """

    def __init__(
        self,
        condition: Callable[..., Any],
        description: str | None = None,
        a_repr: reprlib.Repr = aRepr,
        enabled: EnabledInputType | None = None,
        path: str | None = None,
        error: (Callable[..., ExceptionT] | type[ExceptionT] | BaseException)
        | None = None,
        check_on: InvariantCheckEvent = InvariantCheckEvent.CALL,
    ) -> None:
        """
        Initialize a class decorator to establish the invariant on all the public methods.

        :param condition:
            invariant predicate.

            The condition must not be a coroutine function as dunder functions (including ``__init__``)
            of a class can not be async.
        :param description: textual description of the invariant
        :param a_repr: representation instance that defines how the values are represented
        :param enabled:
                The decorator is applied only if this argument is set.

                Otherwise, the condition check is disabled and there is no run-time overhead.

                The default is to always check the condition unless the interpreter runs in optimized mode (``-O`` or
                ``-OO``).
        :param error:
            The error is expected to denote either:

            * A callable. ``error`` is expected to accept a subset of function arguments and return an exception.
              The ``error`` is called on contract violation and the resulting exception is raised.
            * A subclass of ``BaseException`` which is instantiated with the violation message and raised
              on contract violation.
            * An instance of ``BaseException`` that will be raised with the traceback on contract violation.
        :param check_on:
            Specify when to check the invariant.

            * If :py:attr:`InvariantCheckEvent.CALL` is set, the invariant will be checked on all the method calls
              except ``__setattr__``.
            * If :py:attr:`InvariantCheckEvent.SETATTR` is set, the invariant will be checked on ``__setattr__`` calls.
        :return:

        """
        enabled = _resolve_enabled(enabled, path)
        self.enabled = enabled
        self._invariant = None  # type: Invariant | None

        raise NotImplementedError("The invariant decorator is not implemented yet.")

        if error is None:
            pass
        elif isinstance(error, type):
            if not issubclass(error, BaseException):
                raise ValueError(
                    (
                        "The error of the contract is given as a type, "
                        "but the type does not inherit from BaseException: {}"
                    ).format(error)
                )
        else:
            if (
                not inspect.isfunction(error)
                and not inspect.ismethod(error)
                and not isinstance(error, BaseException)
            ):
                raise ValueError(
                    (
                        "The error of the contract must be either a callable (a function or a method), "
                        "a class (subclass of BaseException) or an instance of BaseException, but got: {}"
                    ).format(error)
                )

        location = None  # type: str | None
        tb_stack = traceback.extract_stack(limit=2)[:1]
        if len(tb_stack) > 0:
            frame = tb_stack[0]
            location = "File {}, line {} in {}".format(
                frame.filename, frame.lineno, frame.name
            )

        if inspect.iscoroutinefunction(condition):
            raise ValueError(
                "Async conditions are not possible in invariants as sync methods such as __init__ have to be wrapped."
            )

        self._invariant = Invariant(
            check_on=check_on,
            condition=condition,
            description=description,
            a_repr=a_repr,
            error=error,
            location=location,
        )

        if self._invariant.mandatory_args and self._invariant.mandatory_args != [
            "self"
        ]:
            raise ValueError(
                "Expected an invariant condition with at most an argument 'self', but got: {}".format(
                    self._invariant.condition_args
                )
            )

    def __call__(self, cls: ClassT) -> ClassT:
        """
        Decorate each of the public methods with the invariant.

        Go through the decorator stack of each function and search for a contract checker. If there is one,
        add the invariant to the checker's invariants. If there is no checker in the stack, wrap the function with a
        contract checker.
        """
        assert self._invariant is not None, (
            "self._contract must be set if the contract was enabled."
        )

        if not hasattr(cls, "__invariants__"):
            invariants = []  # type: list[Invariant]
            setattr(cls, "__invariants__", invariants)

            invariants_on_call = []  # type: list[Invariant]
            setattr(cls, "__invariants_on_call__", invariants_on_call)

            invariants_on_setattr = []  # type: list[Invariant]
            setattr(cls, "__invariants_on_setattr__", invariants_on_setattr)
        else:
            invariants = getattr(cls, "__invariants__")
            assert isinstance(invariants, list), (
                "Expected invariants of class {} to be a list, but got: {}".format(
                    cls, type(invariants)
                )
            )

            invariants_on_call = getattr(cls, "__invariants_on_call__")
            assert isinstance(invariants_on_call, list), (
                "Expected invariants on call of class {} to be a list, "
                "but got: {}".format(cls, type(invariants_on_call))
            )

            invariants_on_setattr = getattr(cls, "__invariants_on_setattr__")
            assert isinstance(invariants_on_setattr, list), (
                "Expected invariants on call of class {} to be a list, "
                "but got: {}".format(cls, type(invariants_on_setattr))
            )

        invariants.append(self._invariant)

        if InvariantCheckEvent.CALL in self._invariant.check_on:
            invariants_on_call.append(self._invariant)

        if InvariantCheckEvent.SETATTR in self._invariant.check_on:
            invariants_on_setattr.append(self._invariant)

        _checkers.add_invariant_checks(cls=cls)

        return cls
