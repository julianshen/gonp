package internal

import (
	"testing"
)

// TestAPIStabilization tests the comprehensive API stabilization system for v1.0
func TestAPIStabilization(t *testing.T) {

	t.Run("API version management and compatibility", func(t *testing.T) {
		// Test API version creation and validation
		v1 := NewAPIVersion(1, 0, 0)
		if v1.Major() != 1 || v1.Minor() != 0 || v1.Patch() != 0 {
			t.Error("API version should be created correctly")
		}

		if v1.String() != "1.0.0" {
			t.Errorf("Expected version string '1.0.0', got '%s'", v1.String())
		}

		// Test version compatibility checking
		v1_1 := NewAPIVersion(1, 1, 0)
		v2_0 := NewAPIVersion(2, 0, 0)

		if !v1.IsCompatibleWith(v1_1) {
			t.Error("v1.0.0 should be compatible with v1.1.0 (minor version increase)")
		}

		if v1.IsCompatibleWith(v2_0) {
			t.Error("v1.0.0 should not be compatible with v2.0.0 (major version increase)")
		}

		// Test semantic versioning rules
		if !v1_1.IsBackwardCompatibleWith(v1) {
			t.Error("v1.1.0 should be backward compatible with v1.0.0")
		}

		if v2_0.IsBackwardCompatibleWith(v1) {
			t.Error("v2.0.0 should not be backward compatible with v1.0.0")
		}
	})

	t.Run("API contract definition and validation", func(t *testing.T) {
		// Test API contract creation
		contract := NewAPIContract("gonp", NewAPIVersion(1, 0, 0))
		if contract.Name() != "gonp" {
			t.Errorf("Expected contract name 'gonp', got '%s'", contract.Name())
		}

		// Test function signature registration
		arrayNewFunc := FunctionSignature{
			Name:        "array.New",
			Package:     "array",
			Parameters:  []ParameterInfo{{Name: "data", Type: "interface{}"}, {Name: "shape", Type: "Shape"}},
			ReturnTypes: []ReturnInfo{{Type: "*Array"}, {Type: "error"}},
			Description: "Creates a new array with given data and shape",
		}

		contract.RegisterFunction(arrayNewFunc)

		// Verify function registration
		registered := contract.GetFunction("array.New")
		if registered == nil {
			t.Error("Function should be registered in contract")
		}

		if registered.Name != "array.New" {
			t.Error("Registered function should have correct name")
		}

		if len(registered.Parameters) != 2 {
			t.Error("Function should have 2 parameters")
		}

		// Test type definition registration
		arrayType := TypeDefinition{
			Name:        "Array",
			Package:     "array",
			Kind:        TypeKindStruct,
			Fields:      []FieldInfo{{Name: "data", Type: "Storage"}, {Name: "shape", Type: "Shape"}},
			Methods:     []MethodInfo{{Name: "Len", ReturnType: "int"}, {Name: "Shape", ReturnType: "Shape"}},
			Description: "N-dimensional array structure",
		}

		contract.RegisterType(arrayType)

		// Verify type registration
		registeredType := contract.GetType("Array")
		if registeredType == nil {
			t.Error("Type should be registered in contract")
		}

		if registeredType.Kind != TypeKindStruct {
			t.Error("Type should have correct kind")
		}

		if len(registeredType.Fields) != 2 {
			t.Error("Type should have 2 fields")
		}
	})

	t.Run("Breaking change detection", func(t *testing.T) {
		// Create baseline contract
		baseline := NewAPIContract("gonp", NewAPIVersion(1, 0, 0))

		// Register baseline function
		baseFunc := FunctionSignature{
			Name:        "math.Add",
			Package:     "math",
			Parameters:  []ParameterInfo{{Name: "a", Type: "float64"}, {Name: "b", Type: "float64"}},
			ReturnTypes: []ReturnInfo{{Type: "float64"}},
		}
		baseline.RegisterFunction(baseFunc)

		// Create new contract with breaking change
		newContract := NewAPIContract("gonp", NewAPIVersion(1, 1, 0))

		// Modified function with different signature (breaking change)
		modifiedFunc := FunctionSignature{
			Name:        "math.Add",
			Package:     "math",
			Parameters:  []ParameterInfo{{Name: "a", Type: "int"}, {Name: "b", Type: "int"}}, // Changed types
			ReturnTypes: []ReturnInfo{{Type: "int"}},
		}
		newContract.RegisterFunction(modifiedFunc)

		// Test breaking change detection
		validator := NewBreakingChangeValidator()
		changes := validator.DetectBreakingChanges(baseline, newContract)

		if len(changes) == 0 {
			t.Error("Should detect breaking changes")
		}

		change := changes[0]
		if change.Type != BreakingChangeParameterType {
			t.Errorf("Expected parameter type change, got %s", change.Type)
		}

		if change.Severity != BreakingSeverityMajor {
			t.Error("Parameter type change should be major breaking change")
		}

		if !change.RequiresMajorVersionBump() {
			t.Error("Major breaking change should require major version bump")
		}
	})

	t.Run("Deprecation management and warnings", func(t *testing.T) {
		contract := NewAPIContract("gonp", NewAPIVersion(1, 2, 0))

		// Register function with deprecation
		deprecatedFunc := FunctionSignature{
			Name:         "array.OldCreate",
			Package:      "array",
			Parameters:   []ParameterInfo{{Name: "size", Type: "int"}},
			ReturnTypes:  []ReturnInfo{{Type: "*Array"}},
			IsDeprecated: true,
			DeprecationInfo: DeprecationInfo{
				Version:        NewAPIVersion(1, 1, 0),
				Reason:         "Use array.New instead for better performance",
				Alternative:    "array.New",
				RemovalVersion: NewAPIVersion(2, 0, 0),
			},
		}

		contract.RegisterFunction(deprecatedFunc)

		// Test deprecation detection
		deprecations := contract.GetDeprecatedFunctions()
		if len(deprecations) != 1 {
			t.Errorf("Expected 1 deprecated function, got %d", len(deprecations))
		}

		deprecation := deprecations[0]
		if deprecation.Alternative != "array.New" {
			t.Error("Deprecation should have correct alternative")
		}

		if deprecation.RemovalVersion.String() != "2.0.0" {
			t.Error("Deprecation should have correct removal version")
		}

		// Test deprecation warning generation
		warning := contract.GenerateDeprecationWarning("array.OldCreate")
		if warning == nil {
			t.Error("Should generate deprecation warning")
		}

		if !warning.IsWarning() {
			t.Error("Deprecation should generate warning level message")
		}

		expectedMsg := "Function array.OldCreate is deprecated since v1.1.0: Use array.New instead for better performance"
		if !contains(warning.Message(), expectedMsg) {
			t.Error("Warning should contain deprecation information")
		}
	})

	t.Run("API documentation generation", func(t *testing.T) {
		contract := NewAPIContract("gonp", NewAPIVersion(1, 0, 0))

		// Register comprehensive API elements
		arrayType := TypeDefinition{
			Name:        "Array",
			Package:     "array",
			Kind:        TypeKindStruct,
			Description: "Multi-dimensional array for numerical computing",
			Fields: []FieldInfo{
				{Name: "data", Type: "Storage", Description: "Underlying data storage"},
				{Name: "shape", Type: "Shape", Description: "Array dimensions"},
			},
			Methods: []MethodInfo{
				{Name: "Len", ReturnType: "int", Description: "Returns total number of elements"},
				{Name: "Shape", ReturnType: "Shape", Description: "Returns array shape"},
			},
			Examples: []string{
				"arr := array.New([]float64{1,2,3,4}, Shape{2,2})",
				"length := arr.Len() // returns 4",
			},
		}
		contract.RegisterType(arrayType)

		newFunc := FunctionSignature{
			Name:        "array.New",
			Package:     "array",
			Parameters:  []ParameterInfo{{Name: "data", Type: "interface{}", Description: "Input data slice"}},
			ReturnTypes: []ReturnInfo{{Type: "*Array", Description: "New array instance"}},
			Description: "Creates new array from data slice with automatic shape inference",
			Examples: []string{
				"arr := array.New([]float64{1,2,3,4})",
				"matrix := array.New([][]float64{{1,2},{3,4}})",
			},
		}
		contract.RegisterFunction(newFunc)

		// Test documentation generation
		docGen := NewAPIDocumentationGenerator()

		// Generate markdown documentation
		markdown := docGen.GenerateMarkdown(contract)
		if len(markdown) == 0 {
			t.Error("Should generate markdown documentation")
		}

		// Verify content includes function documentation
		if !contains(markdown, "array.New") {
			t.Error("Documentation should include registered functions")
		}

		if !contains(markdown, "Creates new array from data slice") {
			t.Error("Documentation should include function descriptions")
		}

		// Generate JSON schema
		jsonSchema := docGen.GenerateJSONSchema(contract)
		if len(jsonSchema) == 0 {
			t.Error("Should generate JSON schema")
		}

		// Generate OpenAPI specification
		openAPI := docGen.GenerateOpenAPISpec(contract)
		if openAPI.Version != "1.0.0" {
			t.Error("OpenAPI spec should have correct version")
		}

		if len(openAPI.Paths) == 0 {
			t.Error("OpenAPI spec should include API paths")
		}
	})
}

// TestAPIVersionValidation tests version validation and migration
func TestAPIVersionValidation(t *testing.T) {

	t.Run("Version parsing and validation", func(t *testing.T) {
		// Test valid version parsing
		version, err := ParseAPIVersion("1.2.3")
		if err != nil {
			t.Fatalf("Should parse valid version: %v", err)
		}

		if version.String() != "1.2.3" {
			t.Error("Parsed version should match original")
		}

		// Test invalid version parsing
		invalidVersions := []string{
			"1.2",        // Missing patch
			"1.2.3.4",    // Too many components
			"a.b.c",      // Non-numeric
			"1.2.3-beta", // Pre-release not supported yet
		}

		for _, invalid := range invalidVersions {
			_, err := ParseAPIVersion(invalid)
			if err == nil {
				t.Errorf("Should reject invalid version: %s", invalid)
			}
		}

		// Test version comparison
		v1_0_0 := NewAPIVersion(1, 0, 0)
		v1_0_1 := NewAPIVersion(1, 0, 1)
		v1_1_0 := NewAPIVersion(1, 1, 0)

		if !v1_0_1.IsNewerThan(v1_0_0) {
			t.Error("v1.0.1 should be newer than v1.0.0")
		}

		if !v1_1_0.IsNewerThan(v1_0_1) {
			t.Error("v1.1.0 should be newer than v1.0.1")
		}

		if v1_0_0.IsNewerThan(v1_0_1) {
			t.Error("v1.0.0 should not be newer than v1.0.1")
		}
	})

	t.Run("Migration path validation", func(t *testing.T) {
		migrator := NewAPIMigrator()

		// Define migration from v1.0.0 to v1.1.0
		migration := APIMigration{
			FromVersion: NewAPIVersion(1, 0, 0),
			ToVersion:   NewAPIVersion(1, 1, 0),
			Description: "Add new array creation methods",
			Changes: []MigrationChange{
				{
					Type:        MigrationChangeAddFunction,
					Description: "Added array.NewFromSlice function",
					Impact:      MigrationImpactAdditive,
				},
			},
		}

		migrator.RegisterMigration(migration)

		// Test migration path calculation
		path := migrator.FindMigrationPath(
			NewAPIVersion(1, 0, 0),
			NewAPIVersion(1, 1, 0),
		)

		if len(path) != 1 {
			t.Errorf("Expected 1 migration step, got %d", len(path))
		}

		if path[0].FromVersion.String() != "1.0.0" {
			t.Error("Migration should start from correct version")
		}

		// Test migration validation
		validator := NewMigrationValidator()
		issues := validator.ValidateMigration(migration)

		if len(issues) > 0 {
			t.Errorf("Valid migration should have no issues: %v", issues)
		}

		// Test invalid migration
		invalidMigration := APIMigration{
			FromVersion: NewAPIVersion(1, 1, 0),
			ToVersion:   NewAPIVersion(1, 0, 0), // Backward migration
		}

		issues = validator.ValidateMigration(invalidMigration)
		if len(issues) == 0 {
			t.Error("Backward migration should have validation issues")
		}
	})
}

// TestAPIContractGeneration tests automatic contract generation from code
func TestAPIContractGeneration(t *testing.T) {

	t.Run("Automatic contract generation from package", func(t *testing.T) {
		generator := NewContractGenerator()

		// Test generating contract from internal package
		contract, err := generator.GenerateFromPackage("internal")
		if err != nil {
			t.Fatalf("Should generate contract from package: %v", err)
		}

		if contract.Name() != "internal" {
			t.Error("Generated contract should have correct name")
		}

		// Verify some known types are detected
		shapeType := contract.GetType("Shape")
		if shapeType == nil {
			t.Error("Should detect Shape type")
		}

		// Verify some known functions are detected
		functions := contract.GetAllFunctions()
		if len(functions) == 0 {
			t.Error("Should detect functions in package")
		}

		// Test contract validation
		validator := NewContractValidator()
		violations := validator.ValidateContract(contract)

		// Should have minimal violations for well-formed package
		if len(violations) > 10 {
			t.Errorf("Too many contract violations: %d", len(violations))
		}
	})

	t.Run("Cross-package dependency analysis", func(t *testing.T) {
		analyzer := NewDependencyAnalyzer()

		// Analyze dependencies for internal package
		deps, err := analyzer.AnalyzeDependencies("internal")
		if err != nil {
			t.Fatalf("Should analyze dependencies: %v", err)
		}

		// Should detect standard library dependencies
		hasStdLib := false
		for _, dep := range deps {
			if dep.IsStandardLibrary() {
				hasStdLib = true
				break
			}
		}

		if !hasStdLib {
			t.Error("Should detect standard library dependencies")
		}

		// Test circular dependency detection
		circular := analyzer.DetectCircularDependencies(deps)
		if len(circular) > 0 {
			t.Errorf("Package should not have circular dependencies: %v", circular)
		}

		// Test compatibility analysis
		compatAnalyzer := NewCompatibilityAnalyzer()
		issues := compatAnalyzer.AnalyzeCompatibility(deps)

		// Should have no major compatibility issues
		majorIssues := 0
		for _, issue := range issues {
			if issue.Severity == CompatibilitySeverityMajor {
				majorIssues++
			}
		}

		if majorIssues > 0 {
			t.Errorf("Should not have major compatibility issues: %d", majorIssues)
		}
	})
}

// TestAPIStabilityMetrics tests API stability measurement and reporting
func TestAPIStabilityMetrics(t *testing.T) {

	t.Run("API stability scoring and metrics", func(t *testing.T) {
		// Create baseline and evolved contracts
		baseline := NewAPIContract("gonp", NewAPIVersion(1, 0, 0))
		evolved := NewAPIContract("gonp", NewAPIVersion(1, 1, 0))

		// Add functions to baseline
		baseFunc := FunctionSignature{
			Name:        "math.Add",
			Package:     "math",
			Parameters:  []ParameterInfo{{Name: "a", Type: "float64"}, {Name: "b", Type: "float64"}},
			ReturnTypes: []ReturnInfo{{Type: "float64"}},
		}
		baseline.RegisterFunction(baseFunc)

		// Add same function plus new one to evolved
		evolved.RegisterFunction(baseFunc)
		newFunc := FunctionSignature{
			Name:        "math.Multiply",
			Package:     "math",
			Parameters:  []ParameterInfo{{Name: "a", Type: "float64"}, {Name: "b", Type: "float64"}},
			ReturnTypes: []ReturnInfo{{Type: "float64"}},
		}
		evolved.RegisterFunction(newFunc)

		// Calculate stability metrics
		calculator := NewStabilityMetricsCalculator()
		metrics := calculator.CalculateMetrics(baseline, evolved)

		// Should have high stability score (only additive changes)
		if metrics.StabilityScore < 0.9 {
			t.Errorf("Expected high stability score, got %f", metrics.StabilityScore)
		}

		if metrics.AddedFunctions != 1 {
			t.Errorf("Expected 1 added function, got %d", metrics.AddedFunctions)
		}

		if metrics.RemovedFunctions != 0 {
			t.Errorf("Expected 0 removed functions, got %d", metrics.RemovedFunctions)
		}

		if metrics.ModifiedFunctions != 0 {
			t.Errorf("Expected 0 modified functions, got %d", metrics.ModifiedFunctions)
		}

		// Test metrics reporting
		reporter := NewMetricsReporter()
		report := reporter.GenerateReport(metrics)

		if len(report) == 0 {
			t.Error("Should generate metrics report")
		}

		if !contains(report, "Stability Score") {
			t.Error("Report should include stability score")
		}
	})

	t.Run("API evolution tracking and recommendations", func(t *testing.T) {
		tracker := NewAPIEvolutionTracker()

		// Track evolution from v1.0.0 to v1.2.0
		v1_0 := NewAPIVersion(1, 0, 0)
		v1_1 := NewAPIVersion(1, 1, 0)
		v1_2 := NewAPIVersion(1, 2, 0)

		// Register evolution events
		tracker.RecordEvolution(EvolutionEvent{
			FromVersion: v1_0,
			ToVersion:   v1_1,
			ChangeType:  EvolutionChangeAdditive,
			Impact:      EvolutionImpactLow,
			Description: "Added new utility functions",
		})

		tracker.RecordEvolution(EvolutionEvent{
			FromVersion: v1_1,
			ToVersion:   v1_2,
			ChangeType:  EvolutionChangeModification,
			Impact:      EvolutionImpactMedium,
			Description: "Improved performance of core functions",
		})

		// Test evolution analysis
		analysis := tracker.AnalyzeEvolution(v1_0, v1_2)
		if len(analysis.Events) != 2 {
			t.Errorf("Expected 2 evolution events, got %d", len(analysis.Events))
		}

		if analysis.OverallImpact != EvolutionImpactMedium {
			t.Error("Overall impact should be medium (highest individual impact)")
		}

		// Test recommendations
		recommender := NewEvolutionRecommender()
		recommendations := recommender.GenerateRecommendations(analysis)

		if len(recommendations) == 0 {
			t.Error("Should generate evolution recommendations")
		}

		// Should recommend testing due to modifications
		hasTestingRecommendation := false
		for _, rec := range recommendations {
			if rec.Category == RecommendationCategoryTesting {
				hasTestingRecommendation = true
				break
			}
		}

		if !hasTestingRecommendation {
			t.Error("Should recommend additional testing for modifications")
		}
	})
}

// Helper function for string containment checking
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		(len(s) > len(substr) &&
			(s[:len(substr)] == substr ||
				s[len(s)-len(substr):] == substr ||
				containsInner(s, substr))))
}

func containsInner(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
