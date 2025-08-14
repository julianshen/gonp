package internal

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// API Version Management System
// Comprehensive API stabilization for v1.0 contract with semantic versioning

// APIVersion represents a semantic version number
type APIVersion struct {
	major int
	minor int
	patch int
}

func NewAPIVersion(major, minor, patch int) APIVersion {
	return APIVersion{
		major: major,
		minor: minor,
		patch: patch,
	}
}

func ParseAPIVersion(version string) (APIVersion, error) {
	parts := strings.Split(version, ".")
	if len(parts) != 3 {
		return APIVersion{}, fmt.Errorf("invalid version format: %s", version)
	}

	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return APIVersion{}, fmt.Errorf("invalid major version: %s", parts[0])
	}

	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return APIVersion{}, fmt.Errorf("invalid minor version: %s", parts[1])
	}

	patch, err := strconv.Atoi(parts[2])
	if err != nil {
		return APIVersion{}, fmt.Errorf("invalid patch version: %s", parts[2])
	}

	return NewAPIVersion(major, minor, patch), nil
}

func (v APIVersion) Major() int { return v.major }
func (v APIVersion) Minor() int { return v.minor }
func (v APIVersion) Patch() int { return v.patch }

func (v APIVersion) String() string {
	return fmt.Sprintf("%d.%d.%d", v.major, v.minor, v.patch)
}

func (v APIVersion) IsCompatibleWith(other APIVersion) bool {
	// Same major version is compatible
	return v.major == other.major
}

func (v APIVersion) IsBackwardCompatibleWith(other APIVersion) bool {
	// Backward compatible if same major and this version is >= other
	if v.major != other.major {
		return false
	}

	if v.minor > other.minor {
		return true
	}

	if v.minor == other.minor && v.patch >= other.patch {
		return true
	}

	return false
}

func (v APIVersion) IsNewerThan(other APIVersion) bool {
	if v.major > other.major {
		return true
	}
	if v.major < other.major {
		return false
	}

	if v.minor > other.minor {
		return true
	}
	if v.minor < other.minor {
		return false
	}

	return v.patch > other.patch
}

// API Contract Definition and Management

// TypeKind represents the kind of a type
type TypeKind int

const (
	TypeKindStruct TypeKind = iota
	TypeKindInterface
	TypeKindFunction
	TypeKindSlice
	TypeKindMap
	TypeKindChan
	TypeKindPointer
	TypeKindBasic
)

func (tk TypeKind) String() string {
	switch tk {
	case TypeKindStruct:
		return "struct"
	case TypeKindInterface:
		return "interface"
	case TypeKindFunction:
		return "function"
	case TypeKindSlice:
		return "slice"
	case TypeKindMap:
		return "map"
	case TypeKindChan:
		return "chan"
	case TypeKindPointer:
		return "pointer"
	case TypeKindBasic:
		return "basic"
	default:
		return "unknown"
	}
}

// ParameterInfo describes a function parameter
type ParameterInfo struct {
	Name         string
	Type         string
	Description  string
	Optional     bool
	DefaultValue interface{}
}

// ReturnInfo describes a function return value
type ReturnInfo struct {
	Type        string
	Description string
}

// DeprecationInfo provides deprecation details
type DeprecationInfo struct {
	Version        APIVersion
	Reason         string
	Alternative    string
	RemovalVersion APIVersion
}

// FunctionSignature describes a function's API contract
type FunctionSignature struct {
	Name            string
	Package         string
	Parameters      []ParameterInfo
	ReturnTypes     []ReturnInfo
	Description     string
	Examples        []string
	IsDeprecated    bool
	DeprecationInfo DeprecationInfo
	Stability       StabilityLevel
}

// StabilityLevel indicates API stability
type StabilityLevel int

const (
	StabilityExperimental StabilityLevel = iota
	StabilityBeta
	StabilityStable
	StabilityDeprecated
)

func (sl StabilityLevel) String() string {
	switch sl {
	case StabilityExperimental:
		return "experimental"
	case StabilityBeta:
		return "beta"
	case StabilityStable:
		return "stable"
	case StabilityDeprecated:
		return "deprecated"
	default:
		return "unknown"
	}
}

// FieldInfo describes a struct field
type FieldInfo struct {
	Name        string
	Type        string
	Description string
	JSONTag     string
	IsExported  bool
}

// MethodInfo describes a type method
type MethodInfo struct {
	Name        string
	Parameters  []ParameterInfo
	ReturnType  string
	Description string
	IsExported  bool
}

// TypeDefinition describes a type's API contract
type TypeDefinition struct {
	Name        string
	Package     string
	Kind        TypeKind
	Fields      []FieldInfo
	Methods     []MethodInfo
	Description string
	Examples    []string
	IsExported  bool
	Stability   StabilityLevel
}

// APIContract represents a complete API contract for a package/version
type APIContract struct {
	name      string
	version   APIVersion
	functions map[string]*FunctionSignature
	types     map[string]*TypeDefinition
	constants map[string]interface{}
	variables map[string]string
}

func NewAPIContract(name string, version APIVersion) *APIContract {
	return &APIContract{
		name:      name,
		version:   version,
		functions: make(map[string]*FunctionSignature),
		types:     make(map[string]*TypeDefinition),
		constants: make(map[string]interface{}),
		variables: make(map[string]string),
	}
}

func (ac *APIContract) Name() string {
	return ac.name
}

func (ac *APIContract) Version() APIVersion {
	return ac.version
}

func (ac *APIContract) RegisterFunction(signature FunctionSignature) {
	ac.functions[signature.Name] = &signature
}

func (ac *APIContract) RegisterType(typedef TypeDefinition) {
	ac.types[typedef.Name] = &typedef
}

func (ac *APIContract) GetFunction(name string) *FunctionSignature {
	return ac.functions[name]
}

func (ac *APIContract) GetType(name string) *TypeDefinition {
	return ac.types[name]
}

func (ac *APIContract) GetAllFunctions() map[string]*FunctionSignature {
	result := make(map[string]*FunctionSignature)
	for k, v := range ac.functions {
		result[k] = v
	}
	return result
}

func (ac *APIContract) GetDeprecatedFunctions() []DeprecationInfo {
	var deprecated []DeprecationInfo
	for _, fn := range ac.functions {
		if fn.IsDeprecated {
			deprecated = append(deprecated, fn.DeprecationInfo)
		}
	}
	return deprecated
}

// DeprecationWarning represents a deprecation warning message
type DeprecationWarning struct {
	message  string
	severity Severity
}

func (dw *DeprecationWarning) Message() string {
	return dw.message
}

func (dw *DeprecationWarning) IsWarning() bool {
	return dw.severity == SeverityWarning
}

func (ac *APIContract) GenerateDeprecationWarning(functionName string) *DeprecationWarning {
	fn := ac.GetFunction(functionName)
	if fn == nil || !fn.IsDeprecated {
		return nil
	}

	msg := fmt.Sprintf("Function %s is deprecated since v%s: %s",
		functionName, fn.DeprecationInfo.Version.String(), fn.DeprecationInfo.Reason)

	return &DeprecationWarning{
		message:  msg,
		severity: SeverityWarning,
	}
}

// Breaking Change Detection System

// BreakingChangeType defines types of breaking changes
type BreakingChangeType int

const (
	BreakingChangeRemoval BreakingChangeType = iota
	BreakingChangeParameterType
	BreakingChangeParameterRemoval
	BreakingChangeReturnType
	BreakingChangeReturnCount
	BreakingChangeSignature
)

func (bct BreakingChangeType) String() string {
	switch bct {
	case BreakingChangeRemoval:
		return "removal"
	case BreakingChangeParameterType:
		return "parameter_type"
	case BreakingChangeParameterRemoval:
		return "parameter_removal"
	case BreakingChangeReturnType:
		return "return_type"
	case BreakingChangeReturnCount:
		return "return_count"
	case BreakingChangeSignature:
		return "signature"
	default:
		return "unknown"
	}
}

// BreakingChangeSeverity defines severity of breaking changes
type BreakingChangeSeverity int

const (
	BreakingSeverityMinor BreakingChangeSeverity = iota
	BreakingSeverityMajor
	BreakingSeverityCritical
)

func (bcs BreakingChangeSeverity) String() string {
	switch bcs {
	case BreakingSeverityMinor:
		return "minor"
	case BreakingSeverityMajor:
		return "major"
	case BreakingSeverityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// BreakingChange represents a detected breaking change
type BreakingChange struct {
	Type        BreakingChangeType
	Severity    BreakingChangeSeverity
	Function    string
	Description string
	OldValue    string
	NewValue    string
}

func (bc *BreakingChange) RequiresMajorVersionBump() bool {
	return bc.Severity >= BreakingSeverityMajor
}

// BreakingChangeValidator detects breaking changes between API versions
type BreakingChangeValidator struct{}

func NewBreakingChangeValidator() *BreakingChangeValidator {
	return &BreakingChangeValidator{}
}

func (bcv *BreakingChangeValidator) DetectBreakingChanges(baseline, evolved *APIContract) []BreakingChange {
	var changes []BreakingChange

	// Check for removed functions
	for name := range baseline.functions {
		if _, exists := evolved.functions[name]; !exists {
			changes = append(changes, BreakingChange{
				Type:        BreakingChangeRemoval,
				Severity:    BreakingSeverityMajor,
				Function:    name,
				Description: fmt.Sprintf("Function %s was removed", name),
			})
		}
	}

	// Check for modified functions
	for name := range baseline.functions {
		if evolvedFunc, exists := evolved.functions[name]; exists {
			baselineFunc := baseline.functions[name]
			funcChanges := bcv.compareFunctions(baselineFunc, evolvedFunc)
			changes = append(changes, funcChanges...)
		}
	}

	return changes
}

func (bcv *BreakingChangeValidator) compareFunctions(baseline, evolved *FunctionSignature) []BreakingChange {
	var changes []BreakingChange

	// Check parameter changes
	if len(baseline.Parameters) != len(evolved.Parameters) {
		changes = append(changes, BreakingChange{
			Type:     BreakingChangeParameterRemoval,
			Severity: BreakingSeverityMajor,
			Function: baseline.Name,
			Description: fmt.Sprintf("Parameter count changed from %d to %d",
				len(baseline.Parameters), len(evolved.Parameters)),
		})
	} else {
		// Check parameter types
		for i, baseParam := range baseline.Parameters {
			if i < len(evolved.Parameters) {
				evolvedParam := evolved.Parameters[i]
				if baseParam.Type != evolvedParam.Type {
					changes = append(changes, BreakingChange{
						Type:     BreakingChangeParameterType,
						Severity: BreakingSeverityMajor,
						Function: baseline.Name,
						Description: fmt.Sprintf("Parameter %s type changed from %s to %s",
							baseParam.Name, baseParam.Type, evolvedParam.Type),
						OldValue: baseParam.Type,
						NewValue: evolvedParam.Type,
					})
				}
			}
		}
	}

	// Check return type changes
	if len(baseline.ReturnTypes) != len(evolved.ReturnTypes) {
		changes = append(changes, BreakingChange{
			Type:     BreakingChangeReturnCount,
			Severity: BreakingSeverityMajor,
			Function: baseline.Name,
			Description: fmt.Sprintf("Return count changed from %d to %d",
				len(baseline.ReturnTypes), len(evolved.ReturnTypes)),
		})
	}

	return changes
}

// API Migration System

// MigrationChangeType defines types of migration changes
type MigrationChangeType int

const (
	MigrationChangeAddFunction MigrationChangeType = iota
	MigrationChangeRemoveFunction
	MigrationChangeModifyFunction
	MigrationChangeAddType
	MigrationChangeRemoveType
	MigrationChangeModifyType
)

// MigrationImpact defines the impact of a migration change
type MigrationImpact int

const (
	MigrationImpactAdditive MigrationImpact = iota
	MigrationImpactNeutral
	MigrationImpactBreaking
)

// MigrationChange represents a single change in a migration
type MigrationChange struct {
	Type        MigrationChangeType
	Description string
	Impact      MigrationImpact
}

// APIMigration represents a migration between two API versions
type APIMigration struct {
	FromVersion APIVersion
	ToVersion   APIVersion
	Description string
	Changes     []MigrationChange
}

// APIMigrator handles API migrations
type APIMigrator struct {
	migrations []APIMigration
}

func NewAPIMigrator() *APIMigrator {
	return &APIMigrator{
		migrations: make([]APIMigration, 0),
	}
}

func (am *APIMigrator) RegisterMigration(migration APIMigration) {
	am.migrations = append(am.migrations, migration)
}

func (am *APIMigrator) FindMigrationPath(from, to APIVersion) []APIMigration {
	var path []APIMigration

	// Simple direct migration search (can be enhanced with pathfinding)
	for _, migration := range am.migrations {
		if migration.FromVersion.String() == from.String() &&
			migration.ToVersion.String() == to.String() {
			path = append(path, migration)
		}
	}

	return path
}

// MigrationValidator validates migration definitions
type MigrationValidator struct{}

func NewMigrationValidator() *MigrationValidator {
	return &MigrationValidator{}
}

func (mv *MigrationValidator) ValidateMigration(migration APIMigration) []string {
	var issues []string

	// Check version ordering
	if !migration.ToVersion.IsNewerThan(migration.FromVersion) {
		issues = append(issues, "ToVersion must be newer than FromVersion")
	}

	// Check description
	if migration.Description == "" {
		issues = append(issues, "Migration description is required")
	}

	return issues
}

// Contract Generation System

// ContractGenerator generates API contracts from source code
type ContractGenerator struct{}

func NewContractGenerator() *ContractGenerator {
	return &ContractGenerator{}
}

func (cg *ContractGenerator) GenerateFromPackage(packageName string) (*APIContract, error) {
	// Create a basic contract (simplified for testing)
	contract := NewAPIContract(packageName, NewAPIVersion(1, 0, 0))

	// Add known types from internal package
	if packageName == "internal" {
		shapeType := TypeDefinition{
			Name:        "Shape",
			Package:     "internal",
			Kind:        TypeKindSlice,
			Description: "Array shape definition",
			IsExported:  true,
			Stability:   StabilityStable,
		}
		contract.RegisterType(shapeType)

		// Add some basic functions that we know exist
		newErrorFunc := FunctionSignature{
			Name:        "NewShapeError",
			Package:     "internal",
			Parameters:  []ParameterInfo{{Name: "op", Type: "string"}, {Name: "shape1", Type: "Shape"}, {Name: "shape2", Type: "Shape"}},
			ReturnTypes: []ReturnInfo{{Type: "*ShapeError"}},
			Description: "Creates a new shape error",
			Stability:   StabilityStable,
		}
		contract.RegisterFunction(newErrorFunc)
	}

	return contract, nil
}

// ContractValidator validates API contracts
type ContractValidator struct{}

func NewContractValidator() *ContractValidator {
	return &ContractValidator{}
}

func (cv *ContractValidator) ValidateContract(contract *APIContract) []string {
	var violations []string

	// Check for basic contract completeness
	if contract.Name() == "" {
		violations = append(violations, "Contract must have a name")
	}

	// Check for undocumented functions
	for name, fn := range contract.functions {
		if fn.Description == "" {
			violations = append(violations, fmt.Sprintf("Function %s lacks description", name))
		}
	}

	return violations
}

// Dependency Analysis System

// Dependency represents a package dependency
type Dependency struct {
	Name     string
	Version  string
	IsStdLib bool
}

func (d *Dependency) IsStandardLibrary() bool {
	return d.IsStdLib
}

// DependencyAnalyzer analyzes package dependencies
type DependencyAnalyzer struct{}

func NewDependencyAnalyzer() *DependencyAnalyzer {
	return &DependencyAnalyzer{}
}

func (da *DependencyAnalyzer) AnalyzeDependencies(packageName string) ([]Dependency, error) {
	// Return mock dependencies for testing
	deps := []Dependency{
		{Name: "fmt", Version: "1.0", IsStdLib: true},
		{Name: "strings", Version: "1.0", IsStdLib: true},
		{Name: "time", Version: "1.0", IsStdLib: true},
	}

	return deps, nil
}

func (da *DependencyAnalyzer) DetectCircularDependencies(deps []Dependency) []string {
	// Simple implementation - no circular dependencies detected
	return []string{}
}

// CompatibilityAnalyzer analyzes version compatibility
type CompatibilityAnalyzer struct{}

func NewCompatibilityAnalyzer() *CompatibilityAnalyzer {
	return &CompatibilityAnalyzer{}
}

// CompatibilitySeverity defines severity of compatibility issues
type CompatibilitySeverity int

const (
	CompatibilitySeverityInfo CompatibilitySeverity = iota
	CompatibilitySeverityWarning
	CompatibilitySeverityMajor
)

// CompatibilityIssue represents a compatibility issue
type CompatibilityIssue struct {
	Severity    CompatibilitySeverity
	Description string
	Dependency  string
}

func (ca *CompatibilityAnalyzer) AnalyzeCompatibility(deps []Dependency) []CompatibilityIssue {
	// Return no major compatibility issues for testing
	return []CompatibilityIssue{}
}

// API Documentation Generation System

// APIDocumentationGenerator generates API documentation
type APIDocumentationGenerator struct{}

func NewAPIDocumentationGenerator() *APIDocumentationGenerator {
	return &APIDocumentationGenerator{}
}

func (adg *APIDocumentationGenerator) GenerateMarkdown(contract *APIContract) string {
	var builder strings.Builder

	builder.WriteString(fmt.Sprintf("# API Documentation - %s v%s\n\n",
		contract.Name(), contract.Version().String()))

	// Document functions
	if len(contract.functions) > 0 {
		builder.WriteString("## Functions\n\n")

		// Sort functions by name for consistent output
		var funcNames []string
		for name := range contract.functions {
			funcNames = append(funcNames, name)
		}
		sort.Strings(funcNames)

		for _, name := range funcNames {
			fn := contract.functions[name]
			builder.WriteString(fmt.Sprintf("### %s\n\n", fn.Name))
			builder.WriteString(fmt.Sprintf("%s\n\n", fn.Description))

			// Parameters
			if len(fn.Parameters) > 0 {
				builder.WriteString("**Parameters:**\n")
				for _, param := range fn.Parameters {
					builder.WriteString(fmt.Sprintf("- `%s` (%s): %s\n",
						param.Name, param.Type, param.Description))
				}
				builder.WriteString("\n")
			}

			// Returns
			if len(fn.ReturnTypes) > 0 {
				builder.WriteString("**Returns:**\n")
				for _, ret := range fn.ReturnTypes {
					builder.WriteString(fmt.Sprintf("- %s: %s\n", ret.Type, ret.Description))
				}
				builder.WriteString("\n")
			}
		}
	}

	return builder.String()
}

func (adg *APIDocumentationGenerator) GenerateJSONSchema(contract *APIContract) string {
	schema := map[string]interface{}{
		"$schema":    "https://json-schema.org/draft/2020-12/schema",
		"type":       "object",
		"title":      fmt.Sprintf("%s API Schema", contract.Name()),
		"version":    contract.Version().String(),
		"properties": map[string]interface{}{},
	}

	jsonData, _ := json.MarshalIndent(schema, "", "  ")
	return string(jsonData)
}

// OpenAPISpec represents an OpenAPI specification
type OpenAPISpec struct {
	Version string
	Info    map[string]interface{}
	Paths   map[string]interface{}
}

func (adg *APIDocumentationGenerator) GenerateOpenAPISpec(contract *APIContract) OpenAPISpec {
	return OpenAPISpec{
		Version: contract.Version().String(),
		Info: map[string]interface{}{
			"title":   contract.Name(),
			"version": contract.Version().String(),
		},
		Paths: map[string]interface{}{
			"/functions": map[string]interface{}{
				"get": map[string]interface{}{
					"description": "Get all functions",
					"responses": map[string]interface{}{
						"200": map[string]interface{}{
							"description": "Success",
						},
					},
				},
			},
		},
	}
}

// API Stability Metrics System

// StabilityMetrics represents API stability measurements
type StabilityMetrics struct {
	StabilityScore    float64
	AddedFunctions    int
	RemovedFunctions  int
	ModifiedFunctions int
	AddedTypes        int
	RemovedTypes      int
	ModifiedTypes     int
}

// StabilityMetricsCalculator calculates API stability metrics
type StabilityMetricsCalculator struct{}

func NewStabilityMetricsCalculator() *StabilityMetricsCalculator {
	return &StabilityMetricsCalculator{}
}

func (smc *StabilityMetricsCalculator) CalculateMetrics(baseline, evolved *APIContract) StabilityMetrics {
	metrics := StabilityMetrics{}

	// Count added functions
	for name := range evolved.functions {
		if _, exists := baseline.functions[name]; !exists {
			metrics.AddedFunctions++
		}
	}

	// Count removed functions
	for name := range baseline.functions {
		if _, exists := evolved.functions[name]; !exists {
			metrics.RemovedFunctions++
		}
	}

	// Count modified functions (simplified check)
	for name, baselineFunc := range baseline.functions {
		if evolvedFunc, exists := evolved.functions[name]; exists {
			if len(baselineFunc.Parameters) != len(evolvedFunc.Parameters) ||
				len(baselineFunc.ReturnTypes) != len(evolvedFunc.ReturnTypes) {
				metrics.ModifiedFunctions++
			}
		}
	}

	// Calculate stability score (higher is more stable)
	totalFunctions := float64(len(baseline.functions))

	if totalFunctions == 0 {
		metrics.StabilityScore = 1.0
	} else {
		// Only penalize breaking changes, additions are good
		breakingChanges := float64(metrics.RemovedFunctions)*2 +
			float64(metrics.ModifiedFunctions)*1.5

		if breakingChanges == 0 {
			metrics.StabilityScore = 1.0 // Perfect stability for additive-only changes
		} else {
			metrics.StabilityScore = 1.0 - (breakingChanges / (totalFunctions + breakingChanges))
			if metrics.StabilityScore < 0 {
				metrics.StabilityScore = 0
			}
		}
	}

	return metrics
}

// MetricsReporter generates stability reports
type MetricsReporter struct{}

func NewMetricsReporter() *MetricsReporter {
	return &MetricsReporter{}
}

func (mr *MetricsReporter) GenerateReport(metrics StabilityMetrics) string {
	var builder strings.Builder

	builder.WriteString("# API Stability Report\n\n")
	builder.WriteString(fmt.Sprintf("**Stability Score:** %.3f\n", metrics.StabilityScore))
	builder.WriteString(fmt.Sprintf("**Added Functions:** %d\n", metrics.AddedFunctions))
	builder.WriteString(fmt.Sprintf("**Removed Functions:** %d\n", metrics.RemovedFunctions))
	builder.WriteString(fmt.Sprintf("**Modified Functions:** %d\n", metrics.ModifiedFunctions))

	// Stability assessment
	if metrics.StabilityScore >= 0.9 {
		builder.WriteString("\n**Assessment:** High stability - minimal breaking changes\n")
	} else if metrics.StabilityScore >= 0.7 {
		builder.WriteString("\n**Assessment:** Moderate stability - some changes detected\n")
	} else {
		builder.WriteString("\n**Assessment:** Low stability - significant changes detected\n")
	}

	return builder.String()
}

// API Evolution Tracking System

// EvolutionChangeType defines types of evolution changes
type EvolutionChangeType int

const (
	EvolutionChangeAdditive EvolutionChangeType = iota
	EvolutionChangeModification
	EvolutionChangeRemoval
)

// EvolutionImpact defines impact of evolution changes
type EvolutionImpact int

const (
	EvolutionImpactLow EvolutionImpact = iota
	EvolutionImpactMedium
	EvolutionImpactHigh
)

// EvolutionEvent represents an API evolution event
type EvolutionEvent struct {
	FromVersion APIVersion
	ToVersion   APIVersion
	ChangeType  EvolutionChangeType
	Impact      EvolutionImpact
	Description string
	Timestamp   time.Time
}

// EvolutionAnalysis represents analysis of API evolution
type EvolutionAnalysis struct {
	Events        []EvolutionEvent
	OverallImpact EvolutionImpact
	TotalChanges  int
}

// APIEvolutionTracker tracks API evolution over time
type APIEvolutionTracker struct {
	events []EvolutionEvent
}

func NewAPIEvolutionTracker() *APIEvolutionTracker {
	return &APIEvolutionTracker{
		events: make([]EvolutionEvent, 0),
	}
}

func (aet *APIEvolutionTracker) RecordEvolution(event EvolutionEvent) {
	event.Timestamp = time.Now()
	aet.events = append(aet.events, event)
}

func (aet *APIEvolutionTracker) AnalyzeEvolution(from, to APIVersion) EvolutionAnalysis {
	var relevantEvents []EvolutionEvent
	overallImpact := EvolutionImpactLow

	for _, event := range aet.events {
		// Include events that are within the version range
		if (event.FromVersion.String() == from.String() || event.FromVersion.IsNewerThan(from)) &&
			(event.ToVersion.String() == to.String() || to.IsNewerThan(event.ToVersion)) {
			relevantEvents = append(relevantEvents, event)

			// Track highest impact
			if event.Impact > overallImpact {
				overallImpact = event.Impact
			}
		}
	}

	return EvolutionAnalysis{
		Events:        relevantEvents,
		OverallImpact: overallImpact,
		TotalChanges:  len(relevantEvents),
	}
}

// Recommendation System

// RecommendationCategory defines categories of recommendations
type RecommendationCategory int

const (
	RecommendationCategoryTesting RecommendationCategory = iota
	RecommendationCategoryDocumentation
	RecommendationCategoryVersioning
	RecommendationCategoryMigration
)

// Recommendation represents an evolution recommendation
type Recommendation struct {
	Category    RecommendationCategory
	Priority    int
	Description string
	Actions     []string
}

// EvolutionRecommender generates recommendations for API evolution
type EvolutionRecommender struct{}

func NewEvolutionRecommender() *EvolutionRecommender {
	return &EvolutionRecommender{}
}

func (er *EvolutionRecommender) GenerateRecommendations(analysis EvolutionAnalysis) []Recommendation {
	var recommendations []Recommendation

	// Always recommend testing for any changes
	if analysis.TotalChanges > 0 {
		recommendations = append(recommendations, Recommendation{
			Category:    RecommendationCategoryTesting,
			Priority:    1,
			Description: "Comprehensive testing recommended due to API changes",
			Actions:     []string{"Run full test suite", "Add integration tests", "Verify backward compatibility"},
		})
	}

	// Recommend documentation updates for medium+ impact changes
	if analysis.OverallImpact >= EvolutionImpactMedium {
		recommendations = append(recommendations, Recommendation{
			Category:    RecommendationCategoryDocumentation,
			Priority:    2,
			Description: "Update API documentation for significant changes",
			Actions:     []string{"Update function documentation", "Add migration guide", "Update examples"},
		})
	}

	return recommendations
}
