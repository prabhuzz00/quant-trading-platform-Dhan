"""Option chain data fetcher using the Dhan API."""

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionChainFetcher:
    """Fetch and parse option chain data from the Dhan API.

    This class wraps :class:`~src.broker.dhan_broker.DhanBroker` to provide
    structured access to option chain data including expiry lists, per-strike
    call/put details, open interest, Greeks, and implied volatility.

    Parameters
    ----------
    broker:
        A :class:`~src.broker.dhan_broker.DhanBroker` instance (paper-trade
        or live).  The broker must have an active API connection – i.e.
        ``DHAN_CLIENT_ID`` and ``DHAN_ACCESS_TOKEN`` must be set in the
        environment or passed to the broker constructor.

    Examples
    --------
    Typical usage::

        from src.broker.dhan_broker import DhanBroker
        from src.data.option_chain import OptionChainFetcher

        broker = DhanBroker()  # reads credentials from .env
        fetcher = OptionChainFetcher(broker)

        expiries = fetcher.get_expiry_list(under_security_id=13)
        chain = fetcher.get_option_chain(
            under_security_id=13,
            under_exchange_segment="IDX_I",
            expiry=expiries[0],
        )
        atm = fetcher.get_atm_options(chain, spot_price=22_500)
        pcr = fetcher.calculate_pcr(chain)
    """

    # Well-known underlying security IDs on Dhan
    NIFTY50_ID = 13
    BANKNIFTY_ID = 25
    FINNIFTY_ID = 27
    MIDCPNIFTY_ID = 442

    def __init__(self, broker: Any) -> None:
        self.broker = broker

    # ------------------------------------------------------------------ #
    #  Expiry list                                                         #
    # ------------------------------------------------------------------ #

    def get_expiry_list(
        self,
        under_security_id: int,
        under_exchange_segment: str = "IDX_I",
    ) -> list[str]:
        """Return the list of available expiry dates for an underlying.

        Parameters
        ----------
        under_security_id:
            Dhan security ID of the underlying index or equity.
        under_exchange_segment:
            Exchange segment: ``"IDX_I"`` for indices, ``"NSE_EQ"`` for
            equity F&O.

        Returns
        -------
        list[str]
            Sorted list of expiry date strings in ``YYYY-MM-DD`` format.
            Returns an empty list when no API connection is available.
        """
        return self.broker.get_expiry_list(
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
        )

    # ------------------------------------------------------------------ #
    #  Option chain                                                        #
    # ------------------------------------------------------------------ #

    def get_option_chain(
        self,
        under_security_id: int,
        under_exchange_segment: str,
        expiry: str,
    ) -> pd.DataFrame:
        """Fetch and parse the option chain into a flat DataFrame.

        Each row represents one strike price and contains columns for both
        the call and put sides.

        Parameters
        ----------
        under_security_id:
            Dhan security ID of the underlying.
        under_exchange_segment:
            Exchange segment of the underlying.
        expiry:
            Expiry date string in ``YYYY-MM-DD`` format.

        Returns
        -------
        pd.DataFrame
            Columns:

            * ``strike_price`` – strike price
            * ``call_security_id`` – Dhan security ID of the call option
            * ``call_ltp`` – last traded price of the call
            * ``call_oi`` – call open interest
            * ``call_volume`` – call volume
            * ``call_iv`` – call implied volatility (%)
            * ``call_delta``, ``call_theta``, ``call_vega``, ``call_gamma``
            * ``put_security_id`` – Dhan security ID of the put option
            * ``put_ltp``, ``put_oi``, ``put_volume``, ``put_iv``
            * ``put_delta``, ``put_theta``, ``put_vega``, ``put_gamma``

            Returns an empty DataFrame when no data is available.
        """
        raw = self.broker.get_option_chain(
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            expiry=expiry,
        )
        if not raw:
            logger.warning(
                "Empty option chain response for security_id=%s expiry=%s",
                under_security_id,
                expiry,
            )
            return pd.DataFrame()

        oc_records = raw.get("oc_data", [])
        if not oc_records:
            return pd.DataFrame()

        rows = []
        for record in oc_records:
            call = record.get("call_options", {})
            put = record.get("put_options", {})
            rows.append(
                {
                    "strike_price": record.get("strike_price", 0),
                    # Call side
                    "call_security_id": str(call.get("security_id", "")),
                    "call_ltp": float(call.get("last_price", 0.0)),
                    "call_oi": int(call.get("oi", 0)),
                    "call_volume": int(call.get("volume", 0)),
                    "call_iv": float(call.get("iv", 0.0)),
                    "call_delta": float(call.get("delta", 0.0)),
                    "call_theta": float(call.get("theta", 0.0)),
                    "call_vega": float(call.get("vega", 0.0)),
                    "call_gamma": float(call.get("gamma", 0.0)),
                    # Put side
                    "put_security_id": str(put.get("security_id", "")),
                    "put_ltp": float(put.get("last_price", 0.0)),
                    "put_oi": int(put.get("oi", 0)),
                    "put_volume": int(put.get("volume", 0)),
                    "put_iv": float(put.get("iv", 0.0)),
                    "put_delta": float(put.get("delta", 0.0)),
                    "put_theta": float(put.get("theta", 0.0)),
                    "put_vega": float(put.get("vega", 0.0)),
                    "put_gamma": float(put.get("gamma", 0.0)),
                }
            )

        df = pd.DataFrame(rows).sort_values("strike_price").reset_index(drop=True)
        logger.info(
            "Parsed option chain: %d strikes for security_id=%s expiry=%s",
            len(df),
            under_security_id,
            expiry,
        )
        return df

    def get_spot_price(
        self,
        under_security_id: int,
        under_exchange_segment: str,
        expiry: str,
    ) -> float:
        """Return the underlying spot price from the option chain response.

        Parameters
        ----------
        under_security_id, under_exchange_segment, expiry:
            Same as :meth:`get_option_chain`.

        Returns
        -------
        float
            Spot price, or ``0.0`` when unavailable.
        """
        raw = self.broker.get_option_chain(
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            expiry=expiry,
        )
        return float(raw.get("last_price", 0.0))

    # ------------------------------------------------------------------ #
    #  Derived analytics                                                   #
    # ------------------------------------------------------------------ #

    def get_atm_options(
        self, chain: pd.DataFrame, spot_price: float
    ) -> dict:
        """Return the ATM (at-the-money) call and put details.

        The ATM strike is the one whose strike price is closest to the
        *spot_price*.

        Parameters
        ----------
        chain:
            DataFrame produced by :meth:`get_option_chain`.
        spot_price:
            Current underlying price.

        Returns
        -------
        dict
            Keys: ``strike_price``, ``call`` (row dict), ``put`` (row dict).
            Returns an empty dict when *chain* is empty.
        """
        if chain.empty:
            return {}

        idx = (chain["strike_price"] - spot_price).abs().idxmin()
        row = chain.loc[idx]
        return {
            "strike_price": row["strike_price"],
            "call": {
                "security_id": row["call_security_id"],
                "ltp": row["call_ltp"],
                "oi": row["call_oi"],
                "iv": row["call_iv"],
                "delta": row["call_delta"],
                "theta": row["call_theta"],
                "vega": row["call_vega"],
            },
            "put": {
                "security_id": row["put_security_id"],
                "ltp": row["put_ltp"],
                "oi": row["put_oi"],
                "iv": row["put_iv"],
                "delta": row["put_delta"],
                "theta": row["put_theta"],
                "vega": row["put_vega"],
            },
        }

    def get_strikes_near_atm(
        self, chain: pd.DataFrame, spot_price: float, n_strikes: int = 5
    ) -> pd.DataFrame:
        """Return the *n_strikes* strikes on each side of ATM.

        Parameters
        ----------
        chain:
            DataFrame produced by :meth:`get_option_chain`.
        spot_price:
            Current underlying price.
        n_strikes:
            Number of strikes to include on each side of ATM (default 5).

        Returns
        -------
        pd.DataFrame
            Subset of *chain* with ``2 * n_strikes + 1`` rows (or fewer at
            the edges).
        """
        if chain.empty:
            return chain

        atm_idx = int((chain["strike_price"] - spot_price).abs().idxmin())
        lo = max(0, atm_idx - n_strikes)
        hi = min(len(chain) - 1, atm_idx + n_strikes)
        return chain.iloc[lo : hi + 1].reset_index(drop=True)

    def calculate_pcr(self, chain: pd.DataFrame) -> float:
        """Compute the Put-Call Ratio (PCR) by open interest.

        PCR = total put OI / total call OI.  A PCR above 1 is generally
        considered bullish (more put protection being bought); below 1 is
        bearish.

        Parameters
        ----------
        chain:
            DataFrame produced by :meth:`get_option_chain`.

        Returns
        -------
        float
            PCR value, or ``0.0`` when call OI is zero or chain is empty.
        """
        if chain.empty:
            return 0.0

        total_call_oi = chain["call_oi"].sum()
        total_put_oi = chain["put_oi"].sum()

        if total_call_oi == 0:
            return 0.0

        pcr = total_put_oi / total_call_oi
        logger.debug(
            "PCR=%.3f (put_oi=%d, call_oi=%d)", pcr, total_put_oi, total_call_oi
        )
        return pcr

    def get_max_pain(self, chain: pd.DataFrame) -> float:
        """Calculate the max-pain strike price.

        Max Pain is the strike at which option writers (sellers) face the
        minimum total payout.  It is computed as the strike that minimises
        the sum of (in-the-money call value + in-the-money put value) across
        all strikes, weighted by open interest.

        Parameters
        ----------
        chain:
            DataFrame produced by :meth:`get_option_chain`.

        Returns
        -------
        float
            Max-pain strike price, or ``0.0`` when chain is empty.
        """
        if chain.empty:
            return 0.0

        strikes = chain["strike_price"].to_numpy()
        call_oi = chain["call_oi"].to_numpy()
        put_oi = chain["put_oi"].to_numpy()

        # Vectorised pain computation using broadcasting:
        # For each candidate strike (rows), compute pain against all other
        # strikes (columns) and sum across columns.
        #
        # call_pain[i] = sum over j where strikes[j] < strikes[i] of
        #                call_oi[j] * (strikes[i] - strikes[j])
        # put_pain[i]  = sum over j where strikes[j] > strikes[i] of
        #                put_oi[j] * (strikes[j] - strikes[i])
        import numpy as np  # noqa: PLC0415

        diff = strikes[:, None] - strikes[None, :]  # shape (n, n)
        call_pain = (call_oi[None, :] * np.maximum(diff, 0)).sum(axis=1)
        put_pain = (put_oi[None, :] * np.maximum(-diff, 0)).sum(axis=1)
        total_pain = call_pain + put_pain
        max_pain_strike = float(strikes[int(total_pain.argmin())])

        logger.debug("Max pain strike: %.2f", max_pain_strike)
        return max_pain_strike
