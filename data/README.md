# Sales Dataset Documentation

## File: `sales_data.csv`

This file provides transactional sales records across September, October, and November 2024.

### Schema:
- `order_id` (string): Unique order transaction identifier (e.g., `ORD-1001`).
- `date` (string, ISO-8601 `YYYY-MM-DD`): Date of transaction.
- `product` (string): Product name (e.g., `Quantum Laptop`, `CyberPhone Pro`).
- `category` (string): Product category (e.g., `Electronics`, `Furniture`, `Accessories`).
- `units_sold` (integer): Number of units sold in this order.
- `unit_price` (float): Price per unit in USD.
- `revenue` (float): Total revenue for this order (`units_sold * unit_price`).

### Key Historical Benchmark Targets:
- **October 2024 Top Product by Revenue**: `Quantum Laptop` with total revenue of `$68,400.00`.
- **September 2024 Quantum Laptop Revenue**: `$44,400.00`.
- **Growth (Oct vs Sep)**: Absolute change = `+$24,000.00`, Percentage growth = `+54.05%`.
- **Edge cases covered**:
  - `SolarKey Mechanical Keyboard`: Sold only in October (0 units/revenue in September), testing zero-baseline handling.
  - Future/past dates outside Sept-Nov: Returns empty record sets.
