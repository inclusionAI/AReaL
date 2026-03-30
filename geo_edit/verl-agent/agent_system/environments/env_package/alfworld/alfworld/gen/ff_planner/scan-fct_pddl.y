%{
#ifdef YYDEBUG
  extern int yydebug=1;
#endif


#include <stdio.h>
#include <string.h> 
#include "ff.h"
#include "memory.h"
#include "parse.h"

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000000
#endif


#ifndef SCAN_ERR
#define SCAN_ERR
#define DEFINE_EXPECTED            0
#define PROBLEM_EXPECTED           1
#define PROBNAME_EXPECTED          2
#define LBRACKET_EXPECTED          3
#define RBRACKET_EXPECTED          4
#define DOMDEFS_EXPECTED           5
#define REQUIREM_EXPECTED          6
#define TYPEDLIST_EXPECTED         7
#define DOMEXT_EXPECTED            8
#define DOMEXTNAME_EXPECTED        9
#define TYPEDEF_EXPECTED          10
#define CONSTLIST_EXPECTED        11
#define PREDDEF_EXPECTED          12 
#define NAME_EXPECTED             13
#define VARIABLE_EXPECTED         14
#define ACTIONFUNCTOR_EXPECTED    15
#define ATOM_FORMULA_EXPECTED     16
#define EFFECT_DEF_EXPECTED       17
#define NEG_FORMULA_EXPECTED      18
#define NOT_SUPPORTED             19
#define SITUATION_EXPECTED        20
#define SITNAME_EXPECTED          21
#define BDOMAIN_EXPECTED          22
#define BADDOMAIN                 23
#define INIFACTS                  24
#define GOALDEF                   25
#define ADLGOAL                   26
#endif


static char * serrmsg[] = {
  "'define' expected",
  "'problem' expected",
  "problem name expected",
  "'(' expected",
  "')' expected",
  "additional domain definitions expected",
  "requirements (e.g. ':strips') expected",
  "typed list of <%s> expected",
  "domain extension expected",
  "domain to be extented expected",
  "type definition expected",
  "list of constants expected",
  "predicate definition expected",
  "<name> expected",
  "<variable> expected",
  "action functor expected",
  "atomic formula expected",
  "effect definition expected",
  "negated atomic formula expected",
  "requirement %s not supported by this IPP version",  
  "'situation' expected",
  "situation name expected",
  "':domain' expected",
  "this problem needs another domain file",
  "initial facts definition expected",
  "goal definition expected",
  "first order logic expression expected",
  NULL
};


void fcterr( int errno, char *par ) {

/*   sact_err = errno; */

/*   if ( sact_err_par ) { */
/*     free( sact_err_par ); */
/*   } */
/*   if ( par ) { */
/*     sact_err_par = new_Token( strlen(par)+1 ); */
/*     strcpy( sact_err_par, par); */
/*   } else { */
/*     sact_err_par = NULL; */
/*   } */

}


static int sact_err;
static char *sact_err_par = NULL;
static Bool sis_negated = FALSE;

%}


%start file


%union {

  char string[MAX_LENGTH];
  char* pstring;
  ParseExpNode *pParseExpNode;
  PlNode* pPlNode;
  FactList* pFactList;
  TokenList* pTokenList;
  TypedList* pTypedList;

}


%type <pstring> problem_name
%type <pParseExpNode> f_exp
%type <pParseExpNode> ground_f_exp
%type <pPlNode> adl_goal_description
%type <pPlNode> adl_goal_description_star
%type <pPlNode> init_el_plus
%type <pPlNode> init_el
%type <pPlNode> literal_name_plus
%type <pPlNode> literal_name
%type <pTokenList> literal_term
%type <pTokenList> atomic_formula_term
%type <pTokenList> term_star
%type <pstring> term
%type <pTokenList> name_star
%type <pTokenList> atomic_formula_name
%type <pstring> predicate
%type <pTypedList> typed_list_name
%type <pTypedList> typed_list_variable
%type <pTokenList> name_plus

%token DEFINE_TOK
%token PROBLEM_TOK
%token SITUATION_TOK
%token BSITUATION_TOK
%token OBJECTS_TOK
%token BDOMAIN_TOK
%token INIT_TOK
%token GOAL_TOK
%token METRIC_TOK
%token AND_TOK
%token NOT_TOK
%token <string> NAME
%token <string> VARIABLE
%token <string> NUM
%token LE_TOK
%token LEQ_TOK
%token EQ_TOK
%token GEQ_TOK
%token GE_TOK
%token MINUS_TOK
%token AD_TOK
%token MU_TOK
%token DI_TOK
%token FORALL_TOK
%token IMPLY_TOK
%token OR_TOK
%token EXISTS_TOK
%token EITHER_TOK
%token OPEN_PAREN
%token CLOSE_PAREN

%%


/**********************************************************************/
file:
/* empty */
|
problem_definition  file
;


/**********************************************************************/
problem_definition : 
OPEN_PAREN DEFINE_TOK         
{ 
  fcterr( PROBNAME_EXPECTED, NULL ); 
}
problem_name  problem_defs  CLOSE_PAREN                 
{  
  gproblem_name = $4;
  if ( gcmd_line.display_info >= 1 ) {
    printf("\nproblem '%s' defined\n", gproblem_name);
  }
}
;


/**********************************************************************/
problem_name :
OPEN_PAREN  PROBLEM_TOK  NAME  CLOSE_PAREN        
{ 
  $$ = new_Token( strlen($3)+1 );
  strcpy($$, $3);
}
;


/**********************************************************************/
base_domain_name :
OPEN_PAREN  BDOMAIN_TOK  NAME  CLOSE_PAREN
{ 
  if ( SAME != strcmp($3, gdomain_name) ) {
    fcterr( BADDOMAIN, NULL );
    yyerror();
  }
}
;


/**********************************************************************/
problem_defs:
/* empty */
|
objects_def  problem_defs
|
init_def  problem_defs
|
goal_def  problem_defs
|
base_domain_name  problem_defs
|
metric_def problem_defs
;


/**********************************************************************/
objects_def:
OPEN_PAREN  OBJECTS_TOK  typed_list_name  CLOSE_PAREN
{ 
  gparse_objects = $3;
}
;


/**********************************************************************/
init_def:
OPEN_PAREN  INIT_TOK
{
  fcterr( INIFACTS, NULL ); 
}
init_el_plus  CLOSE_PAREN
{
  gorig_initial_facts = new_PlNode(AND);
  gorig_initial_facts->sons = $4;
}
;


/**********************************************************************/
goal_def:
OPEN_PAREN  GOAL_TOK
{ 
  fcterr( GOALDEF, NULL ); 
}
adl_goal_description  CLOSE_PAREN
{
  $4->next = gorig_goal_facts;
  gorig_goal_facts = $4;
}
;


/**********************************************************************/
metric_def:
OPEN_PAREN  METRIC_TOK  NAME ground_f_exp CLOSE_PAREN
{

  if ( gparse_metric != NULL ) {
    printf("\n\ndouble metric specification!\n\n");
    exit( 1 );
  }

  gparse_optimization = malloc(strlen($3) + 1);
  strncpy(gparse_optimization, $3, strlen($3) + 1);
  gparse_metric = $4;

}
;






/**********************************************************************
 * Goal description providing full ADL.
 * RETURNS a tree with the connectives in the nodes and the atomic 
 * predicates in the leafs.
 **********************************************************************/
adl_goal_description:
OPEN_PAREN LE_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_PlNode(COMP);
  $$->comp = LE;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN LEQ_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_PlNode(COMP);
  $$->comp = LEQ;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN EQ_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_PlNode(COMP);
  $$->comp = EQ;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN GEQ_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_PlNode(COMP);
  $$->comp = GEQ;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN GE_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_PlNode(COMP);
  $$->comp = GE;
  $$->lh = $3;
  $$->rh = $4;
}
|
literal_term
{ 
  if ( sis_negated ) {
    $$ = new_PlNode(NOT);
    $$->sons = new_PlNode(ATOM);
    $$->sons->atom = $1;
    sis_negated = FALSE;
  } else {
    $$ = new_PlNode(ATOM);
    $$->atom = $1;
  }
}
|
OPEN_PAREN  AND_TOK  adl_goal_description_star  CLOSE_PAREN
{ 
  $$ = new_PlNode(AND);
  $$->sons = $3;
}
|
OPEN_PAREN  OR_TOK  adl_goal_description_star  CLOSE_PAREN
{ 
  $$ = new_PlNode(OR);
  $$->sons = $3;
}
|
OPEN_PAREN  NOT_TOK  adl_goal_description  CLOSE_PAREN
{ 
  $$ = new_PlNode(NOT);
  $$->sons = $3;
}
|
OPEN_PAREN  IMPLY_TOK  adl_goal_description  adl_goal_description  CLOSE_PAREN
{ 
  PlNode *np = new_PlNode(NOT);
  np->sons = $3;
  np->next = $4;

  $$ = new_PlNode(OR);
  $$->sons = np;
}
|
OPEN_PAREN  EXISTS_TOK 
OPEN_PAREN  typed_list_variable  CLOSE_PAREN 
adl_goal_description  CLOSE_PAREN
{ 

  PlNode *pln;

  pln = new_PlNode(EX);
  pln->parse_vars = $4;

  $$ = pln;
  pln->sons = $6;

}
|
OPEN_PAREN  FORALL_TOK 
OPEN_PAREN  typed_list_variable  CLOSE_PAREN 
adl_goal_description  CLOSE_PAREN
{ 

  PlNode *pln;

  pln = new_PlNode(ALL);
  pln->parse_vars = $4;

  $$ = pln;
  pln->sons = $6;

}
;





/**********************************************************************/
adl_goal_description_star:
/* empty */
{
  $$ = NULL;
}
|
adl_goal_description  adl_goal_description_star
{
  $1->next = $2;
  $$ = $1;
}
;









/**********************************************************************
 * initial state: combined literals and assignments
 **********************************************************************/
init_el_plus:
init_el
{
  $$ = $1;
}
|
init_el init_el_plus
{
   $$ = $1;
   $$->next = $2;
}
;


/**********************************************************************/
init_el:
literal_name
{
  $$ = $1;
}
|
OPEN_PAREN EQ_TOK OPEN_PAREN NAME name_star CLOSE_PAREN NUM CLOSE_PAREN
{
  $$ = new_PlNode( COMP );
  $$->comp = EQ;

  $$->lh = new_ParseExpNode( FHEAD );
  $$->lh->atom = new_TokenList();
  $$->lh->atom->item = new_Token( strlen($4)+1 );
  strcpy( $$->lh->atom->item, $4 );
  $$->lh->atom->next = $5;

  $$->rh = new_ParseExpNode( NUMBER );
  $$->rh->atom = new_TokenList();
  $$->rh->atom->item = new_Token( strlen($7)+1 );
  strcpy( $$->rh->atom->item, $7 );
}
|
OPEN_PAREN EQ_TOK NAME NUM CLOSE_PAREN
{
  $$ = new_PlNode( COMP );
  $$->comp = EQ;

  $$->lh = new_ParseExpNode( FHEAD );
  $$->lh->atom = new_TokenList();
  $$->lh->atom->item = new_Token( strlen($3)+1 );
  strcpy( $$->lh->atom->item, $3 );

  $$->rh = new_ParseExpNode( NUMBER );
  $$->rh->atom = new_TokenList();
  $$->rh->atom->item = new_Token( strlen($4)+1 );
  strcpy( $$->rh->atom->item, $4 );
}
;








/**********************************************************************
 * some expressions used in many different rules
 **********************************************************************/
f_exp:
NUM
{ 
  $$ = new_ParseExpNode( NUMBER );
  $$->atom = new_TokenList();
  $$->atom->item = new_Token( strlen($1)+1 );
  strcpy( $$->atom->item, $1 );
}
|
OPEN_PAREN NAME term_star CLOSE_PAREN
{
  $$ = new_ParseExpNode( FHEAD );
  $$->atom = new_TokenList();
  $$->atom->item = new_Token( strlen($2)+1 );
  strcpy( $$->atom->item, $2 );
  $$->atom->next = $3;
}
|
OPEN_PAREN MINUS_TOK f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( MINUS );
  $$->leftson = $3;
}
|
OPEN_PAREN AD_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( AD );
  $$->leftson = $3;
  $$->rightson = $4;
}
|
OPEN_PAREN MINUS_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( SU );
  $$->leftson = $3;
  $$->rightson = $4;
}
|
OPEN_PAREN MU_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( MU );
  $$->leftson = $3;
  $$->rightson = $4;
}
|
OPEN_PAREN DI_TOK f_exp f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( DI );
  $$->leftson = $3;
  $$->rightson = $4;
}
;


/**********************************************************************/
ground_f_exp:
NUM
{ 
  $$ = new_ParseExpNode( NUMBER );
  $$->atom = new_TokenList();
  $$->atom->item = new_Token( strlen($1)+1 );
  strcpy( $$->atom->item, $1 );
}
|
OPEN_PAREN NAME name_star CLOSE_PAREN
{
  $$ = new_ParseExpNode( FHEAD );
  $$->atom = new_TokenList();
  $$->atom->item = new_Token( strlen($2)+1 );
  strcpy( $$->atom->item, $2 );
  $$->atom->next = $3;
}
|
OPEN_PAREN MINUS_TOK ground_f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( MINUS );
  $$->leftson = $3;
}
|
OPEN_PAREN AD_TOK ground_f_exp ground_f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( AD );
  $$->leftson = $3;
  $$->rightson = $4;
}
|
OPEN_PAREN MINUS_TOK ground_f_exp ground_f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( SU );
  $$->leftson = $3;
  $$->rightson = $4;
}
|
OPEN_PAREN MU_TOK ground_f_exp ground_f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( MU );
  $$->leftson = $3;
  $$->rightson = $4;
}
|
OPEN_PAREN DI_TOK ground_f_exp ground_f_exp CLOSE_PAREN
{
  $$ = new_ParseExpNode( DI );
  $$->leftson = $3;
  $$->rightson = $4;
}
;


/**********************************************************************/
literal_term:
OPEN_PAREN  NOT_TOK  atomic_formula_term  CLOSE_PAREN
{ 
  $$ = $3;
  sis_negated = TRUE;
}
|
atomic_formula_term
{
  $$ = $1;
}
;


/**********************************************************************/
atomic_formula_term:
OPEN_PAREN  predicate  term_star  CLOSE_PAREN
{ 
  $$ = new_TokenList();
  $$->item = $2;
  $$->next = $3;
}
|
OPEN_PAREN  EQ_TOK  term_star  CLOSE_PAREN
{
  $$ = new_TokenList();
  $$->item = new_Token( 5 );
  $$->item = "=";
  $$->next = $3;
}
;


/**********************************************************************/
term_star:
/* empty */
{
  $$ = NULL;
}
|
term  term_star
{
  $$ = new_TokenList();
  $$->item = $1;
  $$->next = $2;
}
;


/**********************************************************************/
term:
NAME
{ 
  $$ = new_Token(strlen($1) + 1);
  strcpy($$, $1);
}
|
VARIABLE
{ 
  $$ = new_Token(strlen($1) + 1);
  strcpy($$, $1);
}
;


/**********************************************************************/
name_plus:
NAME
{
  $$ = new_TokenList();
  $$->item = new_Token(strlen($1) + 1);
  strcpy($$->item, $1);
}
|
NAME  name_plus
{
  $$ = new_TokenList();
  $$->item = new_Token(strlen($1) + 1);
  strcpy($$->item, $1);
  $$->next = $2;
}


/**********************************************************************/
typed_list_name:     /* returns TypedList */
/* empty */
{ $$ = NULL; }
|
NAME  EITHER_TOK  name_plus  CLOSE_PAREN  typed_list_name
{ 
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  $$->type = $3;
  $$->next = $5;
}
|
NAME  MINUS_TOK NAME  typed_list_name   /* end of list for one type */
{
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  $$->type = new_TokenList();
  $$->type->item = new_Token( strlen($3)+1 );
  strcpy( $$->type->item, $3 );
  $$->next = $4;
}
|
NAME  typed_list_name        /* a list element (gets type from next one) */
{
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  if ( $2 ) {/* another element (already typed) is following */
    $$->type = copy_TokenList( $2->type );
  } else {/* no further element - it must be an untyped list */
    $$->type = new_TokenList();
    $$->type->item = new_Token( strlen(STANDARD_TYPE)+1 );
    strcpy( $$->type->item, STANDARD_TYPE );
  }
  $$->next = $2;
}
;


/***********************************************/
typed_list_variable:     /* returns TypedList */
/* empty */
{ $$ = NULL; }
|
VARIABLE  EITHER_TOK  name_plus  CLOSE_PAREN  typed_list_variable
{ 
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  $$->type = $3;
  $$->next = $5;
}
|
VARIABLE  MINUS_TOK NAME  typed_list_variable   /* end of list for one type */
{
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  $$->type = new_TokenList();
  $$->type->item = new_Token( strlen($3)+1 );
  strcpy( $$->type->item, $3 );
  $$->next = $4;
}
|
VARIABLE  typed_list_variable        /* a list element (gets type from next one) */
{
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  if ( $2 ) {/* another element (already typed) is following */
    $$->type = copy_TokenList( $2->type );
  } else {/* no further element - it must be an untyped list */
    $$->type = new_TokenList();
    $$->type->item = new_Token( strlen(STANDARD_TYPE)+1 );
    strcpy( $$->type->item, STANDARD_TYPE );
  }
  $$->next = $2;
}
;




/**********************************************************************/
predicate:
NAME
{ 
  $$ = new_Token(strlen($1) + 1);
  strcpy($$, $1);
}
;


/**********************************************************************/
literal_name_plus:
literal_name
{
  $$ = $1;
}
|
literal_name literal_name_plus
{
   $$ = $1;
   $$->next = $2;
}

 
/**********************************************************************/
literal_name:
OPEN_PAREN  NOT_TOK  atomic_formula_name  CLOSE_PAREN
{ 
  PlNode *tmp;

  tmp = new_PlNode(ATOM);
  tmp->atom = $3;
  $$ = new_PlNode(NOT);
  $$->sons = tmp;
}
|
atomic_formula_name
{
  $$ = new_PlNode(ATOM);
  $$->atom = $1;
}
;


/**********************************************************************/
atomic_formula_name:
OPEN_PAREN  predicate  name_star  CLOSE_PAREN
{ 
  $$ = new_TokenList();
  $$->item = $2;
  $$->next = $3;
}
;


/**********************************************************************/
name_star:
/* empty */
{ $$ = NULL; }
|
NAME  name_star
{
  $$ = new_TokenList();
  $$->item = new_Token(strlen($1) + 1);
  strcpy($$->item, $1);
  $$->next = $2;
}
;


%%


#include "lex.fct_pddl.c"


/**********************************************************************
 * Functions
 **********************************************************************/


/* 
 * call	bison -pfct -bscan-fct scan-fct.y
 */



int yyerror( char *msg )

{
  fflush( stdout );
  fprintf(stderr,"\n%s: syntax error in line %d, '%s':\n", 
	  gact_filename, lineno, yytext );

  if ( sact_err_par ) {
    fprintf(stderr, "%s%s\n", serrmsg[sact_err], sact_err_par );
  } else {
    fprintf(stderr,"%s\n", serrmsg[sact_err] );
  }

  exit( 1 );

}



void load_fct_file( char *filename ) 

{

  FILE *fp;/* pointer to input files */
  char tmp[MAX_LENGTH] = "";

  /* open fact file 
   */
  if( ( fp = fopen( filename, "r" ) ) == NULL ) {
    sprintf(tmp, "\nff: can't find fact file: %s\n\n", filename );
    perror(tmp);
    exit ( 1 );
  }

  gact_filename = filename;
  lineno = 1; 
  yyin = fp;

  yyparse();

  fclose( fp );/* and close file again */

}

