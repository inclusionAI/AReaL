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
#define DOMDEF_EXPECTED            0
#define DOMAIN_EXPECTED            1
#define DOMNAME_EXPECTED           2
#define LBRACKET_EXPECTED          3
#define RBRACKET_EXPECTED          4
#define DOMDEFS_EXPECTED           5
#define REQUIREM_EXPECTED          6
#define TYPEDLIST_EXPECTED         7
#define LITERAL_EXPECTED           8
#define PRECONDDEF_UNCORRECT       9
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
#define ACTION                    20
#endif


#define NAME_STR "name\0"
#define VARIABLE_STR "variable\0"
#define STANDARD_TYPE "OBJECT\0"
 

static char *serrmsg[] = {
  "domain definition expected",
  "'domain' expected",
  "domain name expected",
  "'(' expected",
  "')' expected",
  "additional domain definitions expected",
  "requirements (e.g. ':STRIPS') expected",
  "typed list of <%s> expected",
  "literal expected",
  "uncorrect precondition definition",
  "type definition expected",
  "list of constants expected",
  "predicate definition expected",
  "<name> expected",
  "<variable> expected",
  "action functor expected",
  "atomic formula expected",
  "effect definition expected",
  "negated atomic formula expected",
  "requirement %s not supported by this FF version",  
  "action definition is not correct",
  NULL
};


/* void opserr( int errno, char *par ); */
void opserr( int errno, char *par )

{

/*   sact_err = errno; */

/*   if ( sact_err_par ) { */
/*     free(sact_err_par); */
/*   } */
/*   if ( par ) { */
/*     sact_err_par = new_Token(strlen(par)+1); */
/*     strcpy(sact_err_par, par); */
/*   } else { */
/*     sact_err_par = NULL; */
/*   } */

}


static int sact_err;
static char *sact_err_par = NULL;
static PlOperator *scur_op = NULL;
static Bool sis_negated = FALSE;


int supported( char *str )

{

  int i;
  char * sup[] = { ":STRIPS", ":NEGATION", ":EQUALITY",":TYPING", 
		   ":CONDITIONAL-EFFECTS", ":NEGATIVE-PRECONDITIONS", ":DISJUNCTIVE-PRECONDITIONS", 
		   ":EXISTENTIAL-PRECONDITIONS", ":UNIVERSAL-PRECONDITIONS", 
		   ":QUANTIFIED-PRECONDITIONS", ":ADL", ":FLUENTS", ":ACTION-COSTS", ":DERIVED-PREDICATES",
		   NULL };     

  for (i=0; NULL != sup[i]; i++) {
    if ( SAME == strcmp(sup[i], str) ) {
      return TRUE;
    }
  }
  
  return FALSE;

}

%}


%start file


%union {

  char string[MAX_LENGTH];
  char *pstring;
  ParseExpNode *pParseExpNode;
  PlNode *pPlNode;
  FactList *pFactList;
  TokenList *pTokenList;
  TypedList *pTypedList;

}


%type <pPlNode> adl_effect
%type <pPlNode> adl_effect_star
%type <pPlNode> adl_goal_description
%type <pPlNode> adl_goal_description_star
%type <pParseExpNode> f_head
%type <pParseExpNode> f_exp
%type <pTokenList> literal_term
%type <pTokenList> term_star
%type <pTypedList> typed_list_name
%type <pTypedList> typed_list_variable
%type <pstring> term
%type <pTokenList> atomic_formula_term
%type <pTokenList> name_plus
%type <pstring> predicate

%token DEFINE_TOK
%token DOMAIN_TOK
%token REQUIREMENTS_TOK
%token TYPES_TOK
%token NUMBER_TOK
%token EITHER_TOK
%token CONSTANTS_TOK
%token PREDICATES_TOK
%token FUNCTIONS_TOK
%token ACTION_TOK
%token AXIOM_TOK
%token VARS_TOK
%token IMPLIES_TOK
%token PRECONDITION_TOK
%token PARAMETERS_TOK
%token EFFECT_TOK
%token AND_TOK
%token NOT_TOK
%token WHEN_TOK
%token FORALL_TOK
%token IMPLY_TOK
%token OR_TOK
%token EXISTS_TOK
%token LE_TOK
%token LEQ_TOK
%token EQ_TOK
%token GEQ_TOK
%token GE_TOK
%token MINUS_TOK
%token AD_TOK
%token MU_TOK
%token DI_TOK
%token ASSIGN_TOK
%token SCALE_UP_TOK
%token SCALE_DOWN_TOK
%token INCREASE_TOK
%token DECREASE_TOK
%token <string> NAME
%token <string> VARIABLE
%token <string> NUM
%token OPEN_PAREN
%token CLOSE_PAREN

%%

/**********************************************************************/
file:
{ 
  opserr( DOMDEF_EXPECTED, NULL ); 
}
domain_definition 
;
/* can be extended to support 'addenda' and similar stuff */


/**********************************************************************/
domain_definition : 
OPEN_PAREN  DEFINE_TOK  domain_name       
{ 
}
optional_domain_defs 
{
  if ( gcmd_line.display_info >= 1 ) {
    printf("\ndomain '%s' defined\n", gdomain_name);
  }
}
;


/**********************************************************************/
domain_name :
OPEN_PAREN  DOMAIN_TOK  NAME  CLOSE_PAREN 
{ 
  gdomain_name = new_Token( strlen($3)+1 );
  strcpy( gdomain_name, $3);
}
;


/**********************************************************************/
optional_domain_defs:
CLOSE_PAREN  /* end of domain */
|
require_def  optional_domain_defs
|
constants_def  optional_domain_defs
|
types_def  optional_domain_defs
|
predicates_def  optional_domain_defs
|
functions_def  optional_domain_defs
|
action_def  optional_domain_defs
|
axiom_def  optional_domain_defs
;


/**********************************************************************/
predicates_def :
OPEN_PAREN PREDICATES_TOK  predicates_list 
{
}
CLOSE_PAREN
{ 
}
;
/**********************************************************************/
predicates_list :
/* empty = finished */
{}
|
OPEN_PAREN  NAME typed_list_variable  CLOSE_PAREN
{

  TypedListList *tll;

  if ( gparse_predicates ) {
    tll = gparse_predicates;
    while ( tll->next ) {
      tll = tll->next;
    }
    tll->next = new_TypedListList();
    tll = tll->next;
  } else {
    tll = new_TypedListList();
    gparse_predicates = tll;
  }

  tll->predicate = new_Token( strlen( $2 ) + 1);
  strcpy( tll->predicate, $2 );

  tll->args = $3;

}
predicates_list
;


/**********************************************************************/
functions_def :
OPEN_PAREN FUNCTIONS_TOK  functions_list 
{
}
CLOSE_PAREN
{ 
}
;
/**********************************************************************/
functions_list :
/* empty = finished */
{}
|
OPEN_PAREN  NAME typed_list_variable  CLOSE_PAREN
{

  TypedListList *tll;

  if ( gparse_functions ) {
    tll = gparse_functions;
    while ( tll->next ) {
      tll = tll->next;
    }
    tll->next = new_TypedListList();
    tll = tll->next;
  } else {
    tll = new_TypedListList();
    gparse_functions = tll;
  }

  tll->predicate = new_Token( strlen( $2 ) + 1);
  strcpy( tll->predicate, $2 );

  tll->args = $3;

}
functions_list
|
OPEN_PAREN  NAME typed_list_variable  CLOSE_PAREN MINUS_TOK NUMBER_TOK
{

  TypedListList *tll;

  if ( gparse_functions ) {
    tll = gparse_functions;
    while ( tll->next ) {
      tll = tll->next;
    }
    tll->next = new_TypedListList();
    tll = tll->next;
  } else {
    tll = new_TypedListList();
    gparse_functions = tll;
  }

  tll->predicate = new_Token( strlen( $2 ) + 1);
  strcpy( tll->predicate, $2 );

  tll->args = $3;

}
functions_list
;


/**********************************************************************/
require_def:
OPEN_PAREN  REQUIREMENTS_TOK 
{ 
  opserr( REQUIREM_EXPECTED, NULL ); 
}
NAME
{ 
  if ( !supported( $4 ) ) {
    opserr( NOT_SUPPORTED, $4 );
    yyerror();
  }
}
require_key_star  CLOSE_PAREN
;


/**********************************************************************/
require_key_star:
/* empty */
|
NAME
{ 
  if ( !supported( $1 ) ) {
    opserr( NOT_SUPPORTED, $1 );
    yyerror();
  }
}
require_key_star
;


/**********************************************************************/
types_def:
OPEN_PAREN  TYPES_TOK
{ 
  opserr( TYPEDEF_EXPECTED, NULL ); 
}
typed_list_name  CLOSE_PAREN
{
  gparse_types = $4;
}
; 


/**********************************************************************/
constants_def:
OPEN_PAREN  CONSTANTS_TOK
{ 
  opserr( CONSTLIST_EXPECTED, NULL ); 
}
typed_list_name  CLOSE_PAREN
{
  gparse_constants = $4;
}
;


/**********************************************************************
 * actions and their optional definitions
 **********************************************************************/
action_def:
OPEN_PAREN  ACTION_TOK  
{ 
  opserr( ACTION, NULL ); 
}  
NAME
{ 
  scur_op = new_PlOperator( $4 );
  scur_op->axiom = FALSE;
}
param_def  action_def_body  CLOSE_PAREN
{
  scur_op->next = gloaded_ops;
  gloaded_ops = scur_op; 
}
;


/**********************************************************************
 * axioms
 **********************************************************************/
axiom_def:
OPEN_PAREN  AXIOM_TOK OPEN_PAREN  predicate  typed_list_variable CLOSE_PAREN adl_goal_description CLOSE_PAREN
{ 
  PlNode *pln;
  TypedList *tyl;
  TokenList *tmp, *prev;

  pln = new_PlNode(ATOM);
  tmp = new_TokenList();
  tmp->item = new_Token( strlen( $4 )+1 ); 
  strcpy( tmp->item, $4 );
  pln->atom = tmp;

  scur_op = new_PlOperator( $4 );
  scur_op->axiom = TRUE;
  scur_op->effects = pln;

  scur_op->preconds = $7;

  scur_op->params = NULL;
  scur_op->parse_params = $5;
  prev = pln->atom;
  for (tyl = scur_op->parse_params; tyl; tyl = tyl->next) {
    /* to be able to distinguish params from :VARS 
     */
    scur_op->number_of_real_params++;

    tmp = new_TokenList();
    tmp->item = new_Token( strlen( tyl->name )+1 ); 
    strcpy( tmp->item, tyl->name );
    prev->next = tmp;
    prev = tmp;
    
  }
  
  scur_op->next = gloaded_ops;
  gloaded_ops = scur_op; 
}
;


/**********************************************************************/
param_def:
/* empty */
{ 
  scur_op->params = NULL; 
}
|
PARAMETERS_TOK  OPEN_PAREN  typed_list_variable  CLOSE_PAREN
{
  TypedList *tl;
  scur_op->parse_params = $3;
  for (tl = scur_op->parse_params; tl; tl = tl->next) {
    /* to be able to distinguish params from :VARS 
     */
    scur_op->number_of_real_params++;
  }
}
;

/**********************************************************************/
action_def_body:
/* empty */
|
VARS_TOK  OPEN_PAREN  typed_list_variable  CLOSE_PAREN  action_def_body
{
  TypedList *tl = NULL;

  /* add vars as parameters 
   */
  if ( scur_op->parse_params ) {
    for( tl = scur_op->parse_params; tl->next; tl = tl->next ) {
      /* empty, get to the end of list 
       */
    }
    tl->next = $3;
    tl = tl->next;
  } else {
    scur_op->parse_params = $3;
  }
}
|
PRECONDITION_TOK  adl_goal_description
{ 
  scur_op->preconds = $2; 
}
action_def_body
|
EFFECT_TOK  adl_effect
{ 
  scur_op->effects = $2; 
}
action_def_body
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
 * effects as allowed in pddl are saved in FF data structures
 * describes everything after the keyword :effect
 *********************************************************************/
adl_effect:
OPEN_PAREN ASSIGN_TOK f_head f_exp CLOSE_PAREN
{
  $$ = new_PlNode( NEF );
  $$->neft = ASSIGN;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN SCALE_UP_TOK f_head f_exp CLOSE_PAREN
{
  $$ = new_PlNode( NEF );
  $$->neft = SCALE_UP;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN SCALE_DOWN_TOK f_head f_exp CLOSE_PAREN
{
  $$ = new_PlNode( NEF );
  $$->neft = SCALE_DOWN;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN INCREASE_TOK f_head f_exp CLOSE_PAREN
{
  $$ = new_PlNode( NEF );
  $$->neft = INCREASE;
  $$->lh = $3;
  $$->rh = $4;
}
|
OPEN_PAREN DECREASE_TOK f_head f_exp CLOSE_PAREN
{
  $$ = new_PlNode( NEF );
  $$->neft = DECREASE;
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
OPEN_PAREN  AND_TOK  adl_effect_star  CLOSE_PAREN
{ 
  $$ = new_PlNode(AND);
  $$->sons = $3;
}
|
OPEN_PAREN  FORALL_TOK 
OPEN_PAREN  typed_list_variable  CLOSE_PAREN 
adl_effect  CLOSE_PAREN
{ 

  PlNode *pln;

  pln = new_PlNode(ALL);
  pln->parse_vars = $4;

  $$ = pln;
  pln->sons = $6;

}
|
OPEN_PAREN  WHEN_TOK  adl_goal_description  adl_effect  CLOSE_PAREN
{
  /* This will be conditional effects in FF representation, but here
   * a formula like (WHEN p q) will be saved as:
   *  [WHEN]
   *  [sons]
   *   /  \
   * [p]  [q]
   * That means, the first son is p, and the second one is q. 
   */
  $$ = new_PlNode(WHEN);
  $3->next = $4;
  $$->sons = $3;
}
;


/**********************************************************************/
adl_effect_star:
{ 
  $$ = NULL; 
}
|
adl_effect  adl_effect_star
{
  $1->next = $2;
  $$ = $1;
}
;


/**********************************************************************
 * some expressions used in many different rules
 **********************************************************************/
f_head:
OPEN_PAREN NAME term_star CLOSE_PAREN
{
  $$ = new_ParseExpNode( FHEAD );
  $$->atom = new_TokenList();
  $$->atom->item = new_Token( strlen($2)+1 );
  strcpy( $$->atom->item, $2 );
  $$->atom->next = $3;
}
;



f_exp:
NUM
{ 
  $$ = new_ParseExpNode( NUMBER );
  $$->atom = new_TokenList();
  $$->atom->item = new_Token( strlen($1)+1 );
  strcpy( $$->atom->item, $1 );
}
|
f_head
{
  $$ = $1;
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
{ $$ = NULL; }
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
  $$ = new_Token( strlen($1)+1 );
  strcpy( $$, $1 );
}
|
VARIABLE
{ 
  $$ = new_Token( strlen($1)+1 );
  strcpy( $$, $1 );
}
;


/**********************************************************************/
name_plus:
NAME
{
  $$ = new_TokenList();
  $$->item = new_Token( strlen($1)+1 );
  strcpy( $$->item, $1 );
}
|
NAME  name_plus
{
  $$ = new_TokenList();
  $$->item = new_Token( strlen($1)+1 );
  strcpy( $$->item, $1 );
  $$->next = $2;
}
;

/**********************************************************************/
predicate:
NAME
{ 
  $$ = new_Token( strlen($1)+1 );
  strcpy( $$, $1 );
}
;


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



%%
#include "lex.ops_pddl.c"


/**********************************************************************
 * Functions
 **********************************************************************/

/* 
 * call	bison -pops -bscan-ops scan-ops.y
 */

  


int yyerror( char *msg )

{

  fflush(stdout);
  fprintf(stderr, "\n%s: syntax error in line %d, '%s':\n", 
	  gact_filename, lineno, yytext);

  if ( NULL != sact_err_par ) {
    fprintf(stderr, "%s %s\n", serrmsg[sact_err], sact_err_par);
  } else {
    fprintf(stderr, "%s\n", serrmsg[sact_err]);
  }

  exit( 1 );

}



void load_ops_file( char *filename )

{

  FILE * fp;/* pointer to input files */
  char tmp[MAX_LENGTH] = "";

  /* open operator file 
   */
  if( ( fp = fopen( filename, "r" ) ) == NULL ) {
    sprintf(tmp, "\nff: can't find operator file: %s\n\n", filename );
    perror(tmp);
    exit( 1 );
  }

  gact_filename = filename;
  lineno = 1; 
  yyin = fp;

  yyparse();

  fclose( fp );/* and close file again */

}
