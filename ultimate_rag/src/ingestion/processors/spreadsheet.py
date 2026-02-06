"""Spreadsheet processor for CSV and Excel files."""
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseProcessor, ProcessedDocument


class SpreadsheetProcessor(BaseProcessor):
    """Processor for CSV and Excel (XLSX) files.
    
    Extracts tabular data and creates:
    1. A summary text for embedding (schema + stats)
    2. Row data for storage in document_rows table
    """
    
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process a spreadsheet file.
        
        Args:
            file_path: Path to the CSV or XLSX file
            
        Returns:
            ProcessedDocument with summary and row data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        
        # Load dataframe based on extension
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported spreadsheet format: {suffix}")
        
        # Create summary text for embedding
        content = self._create_summary(df, file_path.name)
        
        # Convert rows to dicts for storage
        rows = self._dataframe_to_rows(df)
        
        # Extract schema information
        schema = self._extract_schema(df)
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=content,
            metadata={
                "type": "spreadsheet",
                "format": suffix[1:],  # Remove dot
                "source": str(file_path.absolute()),
                "schema": schema,
                "row_count": len(df),
                "column_count": len(df.columns),
            },
            rows=rows,
        )
    
    def _create_summary(self, df: pd.DataFrame, filename: str) -> str:
        """Create a text summary of the dataframe for embedding.
        
        Args:
            df: The dataframe
            filename: Original filename
            
        Returns:
            Human-readable summary
        """
        lines = [
            f"Dataset: {filename}",
            f"Columns: {', '.join(df.columns.tolist())}",
            f"Total rows: {len(df)}",
            "",
            "Column statistics:",
        ]
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column stats
                stats = f"min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}"
                lines.append(f"  - {col} ({dtype}): {stats}, nulls={null_count}")
            elif pd.api.types.is_string_dtype(df[col]):
                # String column stats
                unique = df[col].nunique()
                lines.append(f"  - {col} ({dtype}): {unique} unique values, nulls={null_count}")
            else:
                lines.append(f"  - {col} ({dtype}): nulls={null_count}")
        
        # Add sample rows
        lines.append("")
        lines.append("Sample data (first 3 rows):")
        for _, row in df.head(3).iterrows():
            row_str = ", ".join(f"{k}={v}" for k, v in row.items())
            lines.append(f"  {row_str}")
        
        return "\n".join(lines)
    
    def _dataframe_to_rows(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert dataframe to list of row dictionaries.
        
        Args:
            df: The dataframe
            
        Returns:
            List of row dicts with serializable values
        """
        # Convert to records and handle NaN values
        rows = df.fillna("").to_dict(orient="records")
        
        # Ensure all values are JSON serializable
        for row in rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif hasattr(value, "item"):  # numpy types
                    row[key] = value.item()
        
        return rows
    
    def _extract_schema(self, df: pd.DataFrame) -> dict[str, str]:
        """Extract column schema from dataframe.
        
        Args:
            df: The dataframe
            
        Returns:
            Dict mapping column names to data types
        """
        return {col: str(df[col].dtype) for col in df.columns}
