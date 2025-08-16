package io

import (
	"database/sql"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/julianshen/gonp/dataframe"
	"github.com/julianshen/gonp/internal"
	"github.com/julianshen/gonp/series"
)

// SQLWriteOptions contains options for writing DataFrames to SQL databases
type SQLWriteOptions struct {
	IfExists    string // "fail", "replace", "append"
	CreateTable bool   // Whether to create table if it doesn't exist
	BatchSize   int    // Number of rows to insert per batch
	ChunkSize   int    // Number of rows to process at once
}

// DefaultSQLWriteOptions returns default options for writing to SQL
func DefaultSQLWriteOptions() *SQLWriteOptions {
	return &SQLWriteOptions{
		IfExists:    "fail",
		CreateTable: true,
		BatchSize:   1000,
		ChunkSize:   10000,
	}
}

// SQLPoolOptions contains options for database connection pooling
type SQLPoolOptions struct {
	MaxOpenConns    int           // Maximum number of open connections
	MaxIdleConns    int           // Maximum number of idle connections
	ConnMaxLifetime time.Duration // Maximum lifetime of connections
}

// DefaultSQLPoolOptions returns default connection pool options
func DefaultSQLPoolOptions() *SQLPoolOptions {
	return &SQLPoolOptions{
		MaxOpenConns:    10,
		MaxIdleConns:    5,
		ConnMaxLifetime: 5 * time.Minute,
	}
}

// SQLConnectionPool manages database connection pooling
type SQLConnectionPool struct {
	DB     *sql.DB
	Driver string
}

// ReadSQL reads data from a SQL query into a DataFrame
func ReadSQL(query string, db *sql.DB) (*dataframe.DataFrame, error) {
	ctx := internal.StartProfiler("IO.ReadSQL")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if query == "" {
		return nil, internal.NewValidationErrorWithMsg("ReadSQL", "query cannot be empty")
	}

	if db == nil {
		return nil, internal.NewValidationErrorWithMsg("ReadSQL", "database connection cannot be nil")
	}

	// Execute query
	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %v", err)
	}
	defer rows.Close()

	// Get column information
	columnNames, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get column names: %v", err)
	}

	columnTypes, err := rows.ColumnTypes()
	if err != nil {
		return nil, fmt.Errorf("failed to get column types: %v", err)
	}

	// Prepare data containers
	numCols := len(columnNames)
	columnData := make([][]interface{}, numCols)
	for i := range columnData {
		columnData[i] = make([]interface{}, 0)
	}

	// Scan rows
	for rows.Next() {
		// Create slice of interface{} to hold row values
		values := make([]interface{}, numCols)
		valuePtrs := make([]interface{}, numCols)
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		err := rows.Scan(valuePtrs...)
		if err != nil {
			return nil, fmt.Errorf("failed to scan row: %v", err)
		}

		// Convert and store values
		for i, val := range values {
			convertedVal, err := convertSQLValue(val, columnTypes[i])
			if err != nil {
				return nil, fmt.Errorf("failed to convert value in column %s: %v", columnNames[i], err)
			}
			columnData[i] = append(columnData[i], convertedVal)
		}
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating rows: %v", err)
	}

	// Create series for each column by type detection
	seriesList := make([]*series.Series, numCols)
	for i, colName := range columnNames {
		colData := columnData[i]
		if len(colData) == 0 {
			// Empty column - create empty float series as default
			ser, err := series.FromSlice([]float64{}, nil, colName)
			if err != nil {
				return nil, fmt.Errorf("failed to create empty series for column %s: %v", colName, err)
			}
			seriesList[i] = ser
			continue
		}

		// Determine column type from first non-nil value
		var ser *series.Series
		var err error

		// Find first non-nil value to determine type
		var firstVal interface{}
		for _, val := range colData {
			if val != nil {
				firstVal = val
				break
			}
		}

		if firstVal == nil {
			// All values are nil, create empty string series
			strSlice := make([]interface{}, len(colData))
			for j := range strSlice {
				strSlice[j] = ""
			}
			ser, err = series.FromSlice(strSlice, nil, colName)
		} else {
			// Create typed series based on first value
			switch firstVal.(type) {
			case string:
				// Create interface slice for strings
				ser, err = series.FromSlice(colData, nil, colName)
			case int64:
				// Convert to int64 slice
				intSlice := make([]int64, len(colData))
				for j, val := range colData {
					if val != nil {
						intSlice[j] = val.(int64)
					}
				}
				ser, err = series.FromSlice(intSlice, nil, colName)
			case float64:
				// Convert to float64 slice
				floatSlice := make([]float64, len(colData))
				for j, val := range colData {
					if val != nil {
						floatSlice[j] = val.(float64)
					}
				}
				ser, err = series.FromSlice(floatSlice, nil, colName)
			case bool:
				// Convert to bool slice
				boolSlice := make([]bool, len(colData))
				for j, val := range colData {
					if val != nil {
						boolSlice[j] = val.(bool)
					}
				}
				ser, err = series.FromSlice(boolSlice, nil, colName)
			default:
				// Default to interface{} slice
				ser, err = series.FromSlice(colData, nil, colName)
			}
		}

		if err != nil {
			return nil, fmt.Errorf("failed to create series for column %s: %v", colName, err)
		}
		seriesList[i] = ser
	}

	// Create DataFrame
	df, err := dataframe.FromSeries(seriesList)
	if err != nil {
		return nil, fmt.Errorf("failed to create DataFrame: %v", err)
	}

	internal.DebugVerbose("ReadSQL completed: %d rows, %d columns", df.Len(), len(df.Columns()))
	return df, nil
}

// WriteSQL writes a DataFrame to a SQL table
func WriteSQL(df *dataframe.DataFrame, tableName string, db *sql.DB, options *SQLWriteOptions) error {
	ctx := internal.StartProfiler("IO.WriteSQL")
	defer func() {
		if ctx != nil {
			ctx.EndProfiler()
		}
	}()

	if df == nil {
		return internal.NewValidationErrorWithMsg("WriteSQL", "DataFrame cannot be nil")
	}

	if tableName == "" {
		return internal.NewValidationErrorWithMsg("WriteSQL", "table name cannot be empty")
	}
	// Validate identifier to prevent SQL injection via table name
	if err := validateIdentifier(tableName); err != nil {
		return fmt.Errorf("invalid table name: %w", err)
	}

	if db == nil {
		return internal.NewValidationErrorWithMsg("WriteSQL", "database connection cannot be nil")
	}

	if options == nil {
		options = DefaultSQLWriteOptions()
	}

	// Check if table exists
	tableExists, err := checkTableExists(db, tableName)
	if err != nil {
		return fmt.Errorf("failed to check table existence: %v", err)
	}

	// Handle table existence according to options
	if tableExists {
		switch options.IfExists {
		case "fail":
			return fmt.Errorf("table %s already exists", tableName)
		case "replace":
			// Drop and recreate table using validated identifier
			dropSQL, derr := buildDropTableSQL(tableName)
			if derr != nil {
				return derr
			}
			_, err = db.Exec(dropSQL)
			if err != nil {
				return fmt.Errorf("failed to drop existing table: %v", err)
			}
			tableExists = false
		case "append":
			// Table exists, we'll append to it
		default:
			return fmt.Errorf("invalid IfExists option: %s", options.IfExists)
		}
	}

	// Create table if needed
	if !tableExists && options.CreateTable {
		err = createTableFromDataFrame(db, tableName, df)
		if err != nil {
			return fmt.Errorf("failed to create table: %v", err)
		}
	} else if !tableExists {
		return fmt.Errorf("table %s does not exist and CreateTable is false", tableName)
	}

	// Insert data
	err = insertDataFrameData(db, tableName, df, options)
	if err != nil {
		return fmt.Errorf("failed to insert data: %v", err)
	}

	internal.DebugVerbose("WriteSQL completed: %d rows written to %s", df.Len(), tableName)
	return nil
}

// NewSQLConnectionPool creates a new database connection pool
func NewSQLConnectionPool(driver, dataSourceName string, options *SQLPoolOptions) (*SQLConnectionPool, error) {
	if !isValidConnectionString(driver, dataSourceName) {
		return nil, fmt.Errorf("invalid connection string for driver %s", driver)
	}

	db, err := sql.Open(driver, dataSourceName)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %v", err)
	}

	if options == nil {
		options = DefaultSQLPoolOptions()
	}

	// Configure connection pool
	db.SetMaxOpenConns(options.MaxOpenConns)
	db.SetMaxIdleConns(options.MaxIdleConns)
	db.SetConnMaxLifetime(options.ConnMaxLifetime)

	// Test connection
	err = db.Ping()
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %v", err)
	}

	pool := &SQLConnectionPool{
		DB:     db,
		Driver: driver,
	}

	return pool, nil
}

// Close closes the connection pool
func (p *SQLConnectionPool) Close() error {
	if p.DB != nil {
		return p.DB.Close()
	}
	return nil
}

// ExecuteSQL executes a SQL statement and returns a DataFrame if it's a SELECT
func ExecuteSQL(query string, db *sql.DB) (*dataframe.DataFrame, error) {
	trimmed := strings.TrimSpace(strings.ToUpper(query))
	if strings.HasPrefix(trimmed, "SELECT") {
		return ReadSQL(query, db)
	}

	// For non-SELECT queries, execute and return empty DataFrame
	_, err := db.Exec(query)
	if err != nil {
		return nil, fmt.Errorf("failed to execute SQL: %v", err)
	}

	// Return empty DataFrame to indicate success
	emptySer := series.Empty(internal.Float64, "result")
	return dataframe.FromSeries([]*series.Series{emptySer})
}

// Helper functions

// convertSQLValue converts a SQL value to appropriate Go type
func convertSQLValue(val interface{}, colType *sql.ColumnType) (interface{}, error) {
	if val == nil {
		return nil, nil
	}

	// Handle []byte (common for TEXT/BLOB columns)
	if bytes, ok := val.([]byte); ok {
		return string(bytes), nil
	}

	// Handle time.Time
	if t, ok := val.(time.Time); ok {
		return t.Format(time.RFC3339), nil
	}

	// For numeric types, try to convert appropriately
	dbType := strings.ToUpper(colType.DatabaseTypeName())
	switch dbType {
	case "INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT":
		converted, err := convertToInt64SQL(val)
		return converted, err
	case "REAL", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC":
		converted, err := convertToFloat64SQL(val)
		return converted, err
	case "BOOLEAN", "BOOL":
		converted, err := convertToBoolSQL(val)
		return converted, err
	default:
		// Default to string for unknown types
		return fmt.Sprintf("%v", val), nil
	}
}

// checkTableExists attempts a portable existence check.
// It tries a simple SELECT with a validated identifier, which works on SQLite/Postgres
// and many MySQL setups (ANSI_QUOTES). Otherwise, errors are passed through.
func checkTableExists(db *sql.DB, tableName string) (bool, error) {
	if err := validateIdentifier(tableName); err != nil {
		return false, err
	}
	selSQL := fmt.Sprintf("SELECT 1 FROM %s LIMIT 1", tableName)
	row := db.QueryRow(selSQL)
	var one int
	if err := row.Scan(&one); err != nil {
		// Not found is treated as non-existent; differentiate common messages conservatively
		if strings.Contains(strings.ToLower(err.Error()), "no such table") ||
			strings.Contains(strings.ToLower(err.Error()), "doesn't exist") ||
			strings.Contains(strings.ToLower(err.Error()), "undefined table") {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// createTableFromDataFrame creates a table schema based on DataFrame structure
func createTableFromDataFrame(db *sql.DB, tableName string, df *dataframe.DataFrame) error {
	if err := validateIdentifier(tableName); err != nil {
		return err
	}
	sqlTypes := inferSQLTypes(df)

	var columnDefs []string
	for _, colName := range df.Columns() {
		if err := validateIdentifier(colName); err != nil {
			return fmt.Errorf("invalid column name %q: %w", colName, err)
		}
		sqlType, exists := sqlTypes[colName]
		if !exists {
			sqlType = "TEXT" // Default fallback
		}
		columnDefs = append(columnDefs, fmt.Sprintf("%s %s", colName, sqlType))
	}

	createSQL := fmt.Sprintf("CREATE TABLE %s (%s)", tableName, strings.Join(columnDefs, ", "))

	_, err := db.Exec(createSQL)
	return err
}

// buildDropTableSQL builds a safe DROP TABLE statement for a validated identifier
func buildDropTableSQL(tableName string) (string, error) {
	if err := validateIdentifier(tableName); err != nil {
		return "", err
	}
	return fmt.Sprintf("DROP TABLE %s", tableName), nil
}

var identRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// validateIdentifier enforces a conservative identifier policy to avoid SQL injection
func validateIdentifier(name string) error {
	if !identRe.MatchString(name) {
		return fmt.Errorf("identifier must match %s", identRe.String())
	}
	return nil
}

// inferSQLTypes infers SQL column types from DataFrame
func inferSQLTypes(df *dataframe.DataFrame) map[string]string {
	types := make(map[string]string)

	for _, colName := range df.Columns() {
		col, err := df.GetColumn(colName)
		if err != nil {
			types[colName] = "TEXT"
			continue
		}

		if col.Len() == 0 {
			types[colName] = "TEXT"
			continue
		}

		// Check first non-nil value to determine type
		var sqlType string
		for i := 0; i < col.Len(); i++ {
			val := col.At(i)
			if val == nil {
				continue
			}

			switch val.(type) {
			case int, int32, int64:
				sqlType = "INTEGER"
			case float32, float64:
				sqlType = "REAL"
			case bool:
				sqlType = "BOOLEAN"
			case string:
				sqlType = "TEXT"
			case time.Time:
				sqlType = "DATETIME"
			default:
				sqlType = "TEXT"
			}
			break
		}

		if sqlType == "" {
			sqlType = "TEXT"
		}

		types[colName] = sqlType
	}

	return types
}

// insertDataFrameData inserts DataFrame data into SQL table
func insertDataFrameData(db *sql.DB, tableName string, df *dataframe.DataFrame, options *SQLWriteOptions) error {
	if df.Len() == 0 {
		return nil // Nothing to insert
	}

	columns := df.Columns()
	numCols := len(columns)

	// Prepare placeholders
	placeholders := make([]string, numCols)
	for i := range placeholders {
		placeholders[i] = "?"
	}

	insertSQL := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)",
		tableName,
		strings.Join(columns, ", "),
		strings.Join(placeholders, ", "))

	// Prepare statement
	stmt, err := db.Prepare(insertSQL)
	if err != nil {
		return fmt.Errorf("failed to prepare insert statement: %v", err)
	}
	defer stmt.Close()

	// Insert data in batches
	batchSize := options.BatchSize
	if batchSize <= 0 {
		batchSize = 1000
	}

	for startRow := 0; startRow < df.Len(); startRow += batchSize {
		endRow := startRow + batchSize
		if endRow > df.Len() {
			endRow = df.Len()
		}

		// Begin transaction for batch
		tx, err := db.Begin()
		if err != nil {
			return fmt.Errorf("failed to begin transaction: %v", err)
		}

		txStmt := tx.Stmt(stmt)

		// Insert rows in batch
		for row := startRow; row < endRow; row++ {
			values := make([]interface{}, numCols)
			for col := 0; col < numCols; col++ {
				val, err := df.IAt(row, col)
				if err != nil {
					tx.Rollback()
					return fmt.Errorf("failed to get value at (%d, %d): %v", row, col, err)
				}
				values[col] = val
			}

			_, err = txStmt.Exec(values...)
			if err != nil {
				tx.Rollback()
				return fmt.Errorf("failed to insert row %d: %v", row, err)
			}
		}

		// Commit batch
		err = tx.Commit()
		if err != nil {
			return fmt.Errorf("failed to commit batch: %v", err)
		}
	}

	return nil
}

// isValidConnectionString validates connection strings for different drivers
func isValidConnectionString(driver, connString string) bool {
	if connString == "" {
		return false
	}

	switch driver {
	case "sqlite3":
		// SQLite accepts file paths or :memory:
		return true
	case "postgres", "postgresql":
		// Basic validation for PostgreSQL connection strings
		return strings.Contains(connString, "postgres://") ||
			strings.Contains(connString, "user=") ||
			strings.Contains(connString, "host=")
	case "mysql":
		// Basic validation for MySQL connection strings
		return strings.Contains(connString, "@tcp(") ||
			strings.Contains(connString, "user:") ||
			strings.Contains(connString, "@unix(")
	default:
		return false
	}
}

// convertToInt64SQL converts various types to int64 for SQL operations
func convertToInt64SQL(val interface{}) (int64, error) {
	switch v := val.(type) {
	case int64:
		return v, nil
	case int:
		return int64(v), nil
	case int32:
		return int64(v), nil
	case int16:
		return int64(v), nil
	case int8:
		return int64(v), nil
	case uint64:
		const maxInt64U = ^uint64(0) >> 1
		if v > maxInt64U { // > MaxInt64
			return 0, fmt.Errorf("uint64 %d overflows int64", v)
		}
		return int64(v), nil
	case uint32:
		return int64(v), nil
	case uint16:
		return int64(v), nil
	case uint8:
		return int64(v), nil
	case float64:
		if v > float64(^int64(0)) || v < float64(^int64(0))*-1-1 {
			return 0, fmt.Errorf("float64 %g out of int64 range", v)
		}
		return int64(v), nil
	case float32:
		f := float64(v)
		if f > float64(^int64(0)) || f < float64(^int64(0))*-1-1 {
			return 0, fmt.Errorf("float32 %g out of int64 range", v)
		}
		return int64(v), nil
	case bool:
		if v {
			return 1, nil
		}
		return 0, nil
	case string:
		i, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("cannot convert string '%s' to int64", v)
		}
		return i, nil
	default:
		return 0, fmt.Errorf("cannot convert %T to int64", val)
	}
}

// convertToFloat64SQL converts various types to float64 for SQL operations
func convertToFloat64SQL(val interface{}) (float64, error) {
	switch v := val.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int32:
		return float64(v), nil
	case int16:
		return float64(v), nil
	case int8:
		return float64(v), nil
	case uint64:
		return float64(v), nil
	case uint32:
		return float64(v), nil
	case uint16:
		return float64(v), nil
	case uint8:
		return float64(v), nil
	case bool:
		if v {
			return 1.0, nil
		}
		return 0.0, nil
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil {
			return 0, fmt.Errorf("cannot convert string '%s' to float64", v)
		}
		return f, nil
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", val)
	}
}

// convertToBoolSQL converts various types to bool for SQL operations
func convertToBoolSQL(val interface{}) (bool, error) {
	switch v := val.(type) {
	case bool:
		return v, nil
	case int64, int, int32, int16, int8:
		return reflect.ValueOf(v).Int() != 0, nil
	case uint64, uint32, uint16, uint8:
		return reflect.ValueOf(v).Uint() != 0, nil
	case float64:
		return v != 0.0, nil
	case float32:
		return v != 0.0, nil
	case string:
		lower := strings.ToLower(v)
		return lower == "true" || lower == "1" || lower == "yes", nil
	default:
		return false, fmt.Errorf("cannot convert %T to bool", val)
	}
}
