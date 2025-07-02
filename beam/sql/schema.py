"""
BeamIbis Schema Management

This module provides a Pydantic-based schema class that leverages Ibis's native
schema conversion capabilities to work seamlessly across all supported backends.
"""

from typing import Dict, Any, Union, Optional, get_type_hints, get_origin, get_args
from pydantic import BaseModel, Field
from datetime import datetime, date, time
import ibis
import ibis.expr.datatypes as dt
import json


class BeamIbisSchema(BaseModel):
    """
    Pydantic-based schema class that leverages Ibis's native schema conversion.
    
    Define schemas using normal Pydantic class syntax with field annotations:
    
    Example:
        >>> class ProductSchema(BeamIbisSchema):
        ...     id: int
        ...     name: str
        ...     price: float
        ...     created_at: datetime
        ...     is_active: bool = True
        ... 
        >>> # Convert to Ibis schema (works with all backends)
        >>> ibis_schema = ProductSchema.to_ibis_schema()
        >>> 
        >>> # Use with any backend
        >>> bq_table = beam_ibis.create_table_from_schema(ProductSchema, 'products')
        >>> pg_table = beam_ibis.create_table_from_schema(ProductSchema, 'products')
    """
    
    # Configure Pydantic to allow arbitrary types
    model_config = {"arbitrary_types_allowed": True}
    
    @classmethod
    def to_ibis_schema(cls) -> ibis.Schema:
        """
        Convert to native Ibis schema based on field annotations.
        
        Returns:
            ibis.Schema: Native Ibis schema that works with all backends
        """
        type_hints = get_type_hints(cls)
        ibis_fields = {}
        
        for field_name, field_type in type_hints.items():
            if field_name.startswith('_'):
                continue  # Skip private fields
                
            ibis_type = cls._python_type_to_ibis(field_type)
            ibis_fields[field_name] = ibis_type
        
        return ibis.Schema(ibis_fields)
    
    @classmethod
    def _python_type_to_ibis(cls, python_type) -> dt.DataType:
        """Convert Python type annotations to Ibis data types."""
        
        # Handle Optional[T] -> Union[T, None]
        origin = get_origin(python_type)
        if origin is Union:
            args = get_args(python_type)
            if len(args) == 2 and type(None) in args:
                # This is Optional[T]
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return cls._python_type_to_ibis(non_none_type)
        
        # Basic type mappings
        type_mapping = {
            int: dt.int64,
            float: dt.float64,
            str: dt.string,
            bool: dt.boolean,
            datetime: dt.timestamp,
            date: dt.date,
            time: dt.time,
            bytes: dt.binary,
            dict: dt.json,
            list: dt.json,  # Store lists as JSON
        }
        
        if python_type in type_mapping:
            return type_mapping[python_type]()
        
        # Handle generic types like List[T], Dict[K, V]
        if origin is list:
            return dt.json()  # Store as JSON
        elif origin is dict:
            return dt.json()  # Store as JSON
        
        # Fallback to string for unknown types
        return dt.string()
    
    @classmethod
    def get_field_names(cls) -> list[str]:
        """Get list of field names from class annotations."""
        type_hints = get_type_hints(cls)
        return [name for name in type_hints.keys() if not name.startswith('_')]
    
    @classmethod
    def get_field_type(cls, name: str) -> dt.DataType:
        """
        Get the Ibis type of a specific field.
        
        Args:
            name: Field name
            
        Returns:
            dt.DataType: Ibis data type of the field
        """
        type_hints = get_type_hints(cls)
        if name not in type_hints:
            raise ValueError(f"Field '{name}' not found in schema")
        return cls._python_type_to_ibis(type_hints[name])
    
    @classmethod
    def create_sample_data(cls, num_rows: int = 10) -> list[dict]:
        """
        Create sample data based on the schema.
        
        Args:
            num_rows: Number of sample rows to generate
            
        Returns:
            List of dictionaries with sample data
        """
        import random
        
        try:
            from faker import Faker
            fake = Faker()
        except ImportError:
            # Fallback to simple random data if faker is not available
            fake = None
        
        type_hints = get_type_hints(cls)
        sample_data = []
        
        for i in range(num_rows):
            row = {}
            for field_name, field_type in type_hints.items():
                if field_name.startswith('_'):
                    continue
                    
                # Generate sample data based on type
                if field_type == int:
                    row[field_name] = random.randint(1, 1000)
                elif field_type == float:
                    row[field_name] = round(random.uniform(1.0, 100.0), 2)
                elif field_type == str:
                    if fake and 'email' in field_name.lower():
                        row[field_name] = fake.email()
                    elif fake and 'name' in field_name.lower():
                        row[field_name] = fake.name()
                    else:
                        row[field_name] = f"{field_name}_{i+1}"
                elif field_type == bool:
                    row[field_name] = random.choice([True, False])
                elif field_type == datetime:
                    if fake:
                        row[field_name] = fake.date_time_this_year()
                    else:
                        row[field_name] = datetime.now()
                elif field_type == date:
                    if fake:
                        row[field_name] = fake.date_this_year()
                    else:
                        row[field_name] = datetime.now().date()
                elif field_type == dict:
                    row[field_name] = {"sample": f"data_{i}"}
                else:
                    row[field_name] = f"value_{i}"
                    
            sample_data.append(row)
            
        return sample_data
    
    def __init_subclass__(cls, **kwargs):
        """Automatically called when subclassing to validate the schema."""
        super().__init_subclass__(**kwargs)
        
        # Validate that the schema can be converted to Ibis
        try:
            cls.to_ibis_schema()
        except Exception as e:
            raise ValueError(f"Invalid schema definition in {cls.__name__}: {e}")


# Pre-defined schema examples using the new syntax
class UserSchema(BeamIbisSchema):
    """Standard user profile schema."""
    user_id: str
    email: str
    name: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    metadata: dict = {}


class EventLogSchema(BeamIbisSchema):
    """Standard event logging schema."""
    id: str
    timestamp: datetime
    user_id: str
    event_type: str
    properties: dict
    session_id: str


class TimeSeriesSchema(BeamIbisSchema):
    """Standard time series schema."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: dict = {}


class OrderSchema(BeamIbisSchema):
    """E-commerce order schema."""
    order_id: str
    customer_id: str
    product_id: str
    quantity: int
    unit_price: float
    total_amount: float
    order_date: datetime
    is_paid: bool = False


# Legacy support for the old dictionary-based approach
class LegacyBeamIbisSchema(BaseModel):
    """Legacy schema class for backward compatibility."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    fields: Dict[str, Union[str, dt.DataType]] = Field(
        ..., 
        description="Dictionary mapping column names to Ibis data types"
    )
    description: Optional[str] = Field(
        None, 
        description="Human-readable description of the schema"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the schema"
    )
    
    def to_ibis_schema(self) -> ibis.Schema:
        """Convert to native Ibis schema."""
        return ibis.Schema(self.fields)


def example_usage():
    """Example usage of the new BeamIbisSchema."""
    
    # 1. Define schemas using normal Python class syntax
    class ProductSchema(BeamIbisSchema):
        id: int
        name: str
        price: float
        category: str
        created_at: datetime
        is_available: bool = True
    
    # 2. The schema automatically works with all Ibis backends
    ibis_schema = ProductSchema.to_ibis_schema()
    print("Ibis schema:", ibis_schema)
    
    # 3. Use with BeamIbis (would work with any backend)
    # from beam.sql import beam_ibis
    # 
    # # Works with BigQuery
    # bq = beam_ibis('ibis-bigquery:///my-project/ecommerce')
    # products_table = bq.create_table_from_schema(ProductSchema, 'products')
    # 
    # # Same schema works with PostgreSQL
    # pg = beam_ibis('ibis-postgresql://localhost/ecommerce')
    # products_table_pg = pg.create_table_from_schema(ProductSchema, 'products')
    
    # 4. Generate sample data
    sample_data = ProductSchema.create_sample_data(5)
    print("Sample data:", sample_data[:2])
    
    return ProductSchema


if __name__ == "__main__":
    # Run example
    schema = example_usage()
    print(f"\nSchema: {schema.__name__}")
    print(f"Fields: {schema.get_field_names()}") 