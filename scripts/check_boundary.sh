#!/usr/bin/env bash
# Prove boundary/ (and the rest of the organizer-owned tooling) is byte-for-byte
# what the template ships. The onboarding requires an unmodified boundary/, and
# a local edit there is the one failure that passes every local test and then
# fails on the robot -- mocks/mock_wbc.py imports its constants from the same
# module our client encodes with, so editing both keeps conformance green.
set -euo pipefail
cd "$(dirname "$0")/.."
sha256sum --check --strict docs/boundary.sha256
echo "boundary/ and the organizer's tooling are unmodified."
