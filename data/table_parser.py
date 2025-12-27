"""Table parsing and processing"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class Table:
    """Table data structure"""
    headers: List[str]
    rows: List[List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    table_id: Optional[str] = None

    def __len__(self) -> int:
        """Get number of rows"""
        return len(self.rows)

    def __getitem__(self, idx: int) -> List[str]:
        """Get row by index"""
        return self.rows[idx]

    def get_cell(self, row: int, col: int) -> str:
        """Get cell value"""
        return self.rows[row][col]

    def get_column(self, col: int) -> List[str]:
        """Get column values"""
        return [row[col] for row in self.rows]

    def get_column_by_name(self, name: str) -> List[str]:
        """Get column values by header name"""
        try:
            col_idx = self.headers.index(name)
            return self.get_column(col_idx)
        except ValueError:
            raise ValueError(f"Column '{name}' not found in table")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame(self.rows, columns=self.headers)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'headers': self.headers,
            'rows': self.rows,
            'metadata': self.metadata,
            'table_id': self.table_id
        }


class TableParser:
    """Parse and process tables"""

    def __init__(
        self,
        max_rows: int = 50,
        max_cols: int = 20,
        normalize: bool = True
    ):
        """
        Initialize table parser

        Args:
            max_rows: Maximum number of rows to keep
            max_cols: Maximum number of columns to keep
            normalize: Whether to normalize values
        """
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.normalize = normalize

    def parse(
        self,
        raw_table: Any,
        table_id: Optional[str] = None
    ) -> Table:
        """
        Parse raw table to Table object

        Args:
            raw_table: Raw table data (dict, DataFrame, or list)
            table_id: Optional table identifier

        Returns:
            Table object
        """
        # Convert different formats to standard format
        if isinstance(raw_table, pd.DataFrame):
            headers = raw_table.columns.tolist()
            rows = raw_table.values.tolist()
        elif isinstance(raw_table, dict):
            headers = raw_table.get('headers', raw_table.get('header', []))
            rows = raw_table.get('rows', raw_table.get('data', []))
        elif isinstance(raw_table, list):
            # Assume first row is header
            headers = raw_table[0] if raw_table else []
            rows = raw_table[1:] if len(raw_table) > 1 else []
        else:
            raise ValueError(f"Unsupported table format: {type(raw_table)}")

        # Truncate if too large
        headers = headers[:self.max_cols]
        rows = rows[:self.max_rows]

        # Ensure consistent row length
        rows = [row[:self.max_cols] for row in rows]

        # Convert all cells to strings
        headers = [str(h) for h in headers]
        rows = [[str(cell) for cell in row] for row in rows]

        # Normalize if requested
        if self.normalize:
            rows = self._normalize_table(rows)

        table = Table(
            headers=headers,
            rows=rows,
            metadata={'num_rows': len(rows), 'num_cols': len(headers)},
            table_id=table_id
        )

        return table

    def _normalize_table(self, rows: List[List[str]]) -> List[List[str]]:
        """
        Normalize table values

        Args:
            rows: Table rows

        Returns:
            Normalized rows
        """
        normalized_rows = []
        for row in rows:
            normalized_row = []
            for cell in row:
                # Strip whitespace
                cell = str(cell).strip()

                # Normalize numbers
                try:
                    # Try to parse as number
                    num = float(cell.replace(',', ''))
                    # Format consistently
                    if num.is_integer():
                        cell = str(int(num))
                    else:
                        cell = f"{num:.2f}"
                except:
                    pass

                normalized_row.append(cell)
            normalized_rows.append(normalized_row)

        return normalized_rows

    def linearize(self, table: Table, format: str = "markdown") -> str:
        """
        Linearize table to text

        Args:
            table: Table object
            format: Output format ('markdown', 'simple', 'verbose')

        Returns:
            Linearized table string
        """
        if format == "markdown":
            return self._linearize_markdown(table)
        elif format == "simple":
            return self._linearize_simple(table)
        elif format == "verbose":
            return self._linearize_verbose(table)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _linearize_markdown(self, table: Table) -> str:
        """Linearize as markdown table"""
        lines = []

        # Header
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("|" + "|".join(["---"] * len(table.headers)) + "|")

        # Rows
        for row in table.rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _linearize_simple(self, table: Table) -> str:
        """Linearize as simple text"""
        lines = []

        # Header
        lines.append(" | ".join(table.headers))
        lines.append("-" * 50)

        # Rows
        for row in table.rows:
            lines.append(" | ".join(row))

        return "\n".join(lines)

    def _linearize_verbose(self, table: Table) -> str:
        """Linearize with row/col labels"""
        parts = []

        # Add headers
        parts.append("<headers>")
        for i, header in enumerate(table.headers):
            parts.append(f"col_{i}: {header}")
        parts.append("</headers>")

        # Add rows
        parts.append("<rows>")
        for i, row in enumerate(table.rows):
            parts.append(f"<row_{i}>")
            for j, cell in enumerate(row):
                parts.append(f"{table.headers[j]}: {cell}")
            parts.append(f"</row_{i}>")
        parts.append("</rows>")

        return "\n".join(parts)

    def extract_cells(
        self,
        table: Table,
        positions: List[tuple]
    ) -> List[str]:
        """
        Extract cells at specified positions

        Args:
            table: Table object
            positions: List of (row, col) tuples

        Returns:
            List of cell values
        """
        cells = []
        for row, col in positions:
            if 0 <= row < len(table) and 0 <= col < len(table.headers):
                cells.append(table.get_cell(row, col))
            else:
                cells.append("")
        return cells


def parse_table_from_dict(table_dict: Dict[str, Any]) -> Table:
    """
    Convenience function to parse table from dictionary

    Args:
        table_dict: Dictionary containing table data

    Returns:
        Table object
    """
    parser = TableParser()
    return parser.parse(table_dict)


def parse_table_from_df(df: pd.DataFrame) -> Table:
    """
    Convenience function to parse table from DataFrame

    Args:
        df: Pandas DataFrame

    Returns:
        Table object
    """
    parser = TableParser()
    return parser.parse(df)
