#!/bin/bash
set -e

echo "📌 Running pylint on all tracked Python files..."

pylint $(git ls-files '*.py')
