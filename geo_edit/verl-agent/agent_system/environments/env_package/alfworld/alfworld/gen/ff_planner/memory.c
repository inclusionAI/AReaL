/*********************************************************************
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 *********************************************************************/


/*********************************************************************
 * File: memory.c
 * Description: Creation and Deletion functions for all data structures.
 *
 * Author: Joerg Hoffmann
 *
 *********************************************************************/ 









#include "ff.h"
#include "memory.h"


#include "inst_pre.h"






/**********************
 * CREATION FUNCTIONS *
 **********************/











/* parsing
 */









char *new_Token( int len )

{

  char *tok = ( char * ) calloc( len, sizeof( char ) );
  CHECK_PTR(tok);

  return tok;

}



TokenList *new_TokenList( void )

{

  TokenList *result = ( TokenList * ) calloc( 1, sizeof( TokenList ) );
  CHECK_PTR(result);

  result->item = NULL; 
  result->next = NULL;

  return result;

}



FactList *new_FactList( void )

{

  FactList *result = ( FactList * ) calloc( 1, sizeof( FactList ) );
  CHECK_PTR(result);

  result->item = NULL; 
  result->next = NULL;

  return result;

}



TypedList *new_TypedList( void )

{

  TypedList *result = ( TypedList * ) calloc( 1, sizeof( TypedList ) );
  CHECK_PTR(result);

  result->name = NULL; 
  result->type = NULL;
  result->n = -1;

  return result;

}



TypedListList *new_TypedListList( void )

{

  TypedListList *result = ( TypedListList * ) calloc( 1, sizeof( TypedListList ) );
  CHECK_PTR(result);

  result->predicate = NULL; 
  result->args = NULL;

  return result;

}



ParseExpNode *new_ParseExpNode( ExpConnective c )

{

  ParseExpNode *result = ( ParseExpNode * ) calloc( 1, sizeof( ParseExpNode ) );
  CHECK_PTR(result);

  result->connective = c;
  result->atom = NULL;
  result->leftson = NULL;
  result->rightson = NULL;

  return result;

}



PlNode *new_PlNode( Connective c )

{

  PlNode *result = ( PlNode * ) calloc( 1, sizeof( PlNode ) );
  CHECK_PTR(result);

  result->connective = c;
  result->atom = NULL;

  result->comp = -1;
  result->neft = -1;
  result->lh = NULL;
  result->rh = NULL;

  result->sons = NULL;
  result->next = NULL;

  return result;

}



PlOperator *new_PlOperator( char *name )

{

  PlOperator *result = ( PlOperator * ) calloc( 1, sizeof( PlOperator ) );
  CHECK_PTR(result);

  if ( name ) {
    result->name = new_Token(strlen(name)+1);
    CHECK_PTR(result->name);
    strcpy(result->name, name);
  } else {
    result->name = NULL;
  }

  result->params = NULL;
  result->preconds = NULL;
  result->effects = NULL;
  result->number_of_real_params = 0;
  result->next = NULL;

  return result;

}



PlOperator *new_axiom_op_list( void )

{

  static int count;
  char *name;
  PlOperator *ret;

  /* WARNING: count should not exceed 999 
   */
  count++;
  if ( count == 10000 ) {
    printf("\ntoo many axioms! look into memory.c, line 157\n\n");
    exit( 1 );
  }
  name = new_Token(strlen(HIDDEN_STR)+strlen(AXIOM_STR)+4+1);
  sprintf(name, "%s%s%4d", HIDDEN_STR, AXIOM_STR, count);

  ret = new_PlOperator(name);
  free(name);

  return ret;

}














/* instantiation
 */











Fact *new_Fact( void )

{

  Fact *result = ( Fact * ) calloc( 1, sizeof( Fact ) );
  CHECK_PTR(result);

  return result;

}



Fluent *new_Fluent( void )

{

  Fluent *result = ( Fluent * ) calloc( 1, sizeof( Fluent ) );
  CHECK_PTR(result);

  return result;

}



FluentValue *new_FluentValue( void )

{

  FluentValue *result = ( FluentValue * ) calloc( 1, sizeof( FluentValue ) );
  CHECK_PTR(result);

  return result;

}



Facts *new_Facts( void )

{

  Facts *result = ( Facts * ) calloc( 1, sizeof( Facts ) );
  CHECK_PTR(result);

  result->fact = new_Fact();

  result->next = NULL;

  return result;

}



FluentValues *new_FluentValues( void )

{

  FluentValues *result = ( FluentValues * ) calloc( 1, sizeof( FluentValues ) );
  CHECK_PTR(result);

  result->next = NULL;

  return result;

}



ExpNode *new_ExpNode( ExpConnective c )

{

  ExpNode *result = ( ExpNode * ) calloc( 1, sizeof( ExpNode ) );
  CHECK_PTR(result);

  result->connective = c;
  result->fluent = NULL;
  result->fl = -2;
  result->c = 1;
  result->son = NULL;
  result->leftson = NULL;
  result->rightson = NULL;

  return result;

}



WffNode *new_WffNode( Connective c )

{

  WffNode *result = ( WffNode * ) calloc( 1, sizeof( WffNode ) );
  CHECK_PTR(result);

  result->connective = c;

  result->var = -1;
  result->var_type = -1;
  result->var_name = NULL;

  result->sons = NULL;
  result->next = NULL;
  result->prev = NULL;

  result->fact = NULL;
  result->NOT_p = -1;

  result->son = NULL;

  result->comp = -1;
  result->lh = NULL;
  result->rh = NULL;
  
  result->visited = FALSE;

  return result;

}



Literal *new_Literal( void ) 

{

  Literal *result = ( Literal * ) calloc( 1, sizeof( Literal ) );
  CHECK_PTR(result);

  result->next = NULL;
  result->prev = NULL;

  return result; 

}



NumericEffect *new_NumericEffect( void ) 

{

  NumericEffect *result = ( NumericEffect * ) calloc( 1, sizeof( NumericEffect ) );
  CHECK_PTR(result);

  result->rh = NULL;

  result->next = NULL;
  result->prev = NULL;

  return result; 

}



Effect *new_Effect( void )

{

  Effect *result = ( Effect * ) calloc( 1, sizeof( Effect ) );
  CHECK_PTR(result);

  result->num_vars = 0;

  result->conditions = NULL;

  result->effects = NULL;
  result->numeric_effects = NULL;

  result->next = NULL;
  result->prev = NULL;

  return result;

}



Operator *new_Operator( char *name, int norp )

{

  int i;

  Operator *result = ( Operator * ) calloc( 1, sizeof( Operator ) );
  CHECK_PTR(result);

  if ( name ) {
    result->name = new_Token( strlen( name ) + 1 );
    CHECK_PTR( result->name );
    strcpy( result->name, name );
  } else {
    result->name = NULL;
  }

  result->num_vars = 0;
  result->number_of_real_params = norp;

  for ( i = 0; i < MAX_VARS; i++ ) {
    result->removed[i] = FALSE;
  }

  result->preconds = NULL;

  result->effects = NULL;

  result->hard = TRUE;

  return result;

}



NormEffect *new_NormEffect1( Effect *e )

{

  int i;

  NormEffect *result = ( NormEffect * ) calloc( 1, sizeof( NormEffect ) );
  CHECK_PTR(result);

  result->num_vars = e->num_vars;
  for ( i = 0; i < e->num_vars; i++ ) {
    result->var_types[i] = e->var_types[i];
    result->inst_table[i] = -1;
  }

  result->conditions = NULL;
  result->num_conditions = 0;

  result->adds = NULL;
  result->num_adds = 0;
  result->dels = NULL;
  result->num_dels = 0;

  result->numeric_conditions_comp = NULL;
  result->numeric_conditions_lh = NULL;
  result->numeric_conditions_rh = NULL;
  result->num_numeric_conditions = 0;

  result->numeric_effects_neft = NULL;
  result->numeric_effects_fluent = NULL;
  result->numeric_effects_rh = NULL;
  result->num_numeric_effects = 0;

  result->next = NULL;
  result->prev = NULL;

  return result;

}



NormEffect *new_NormEffect2( NormEffect *e )

{

  int i, j;

  NormEffect *result = ( NormEffect * ) calloc( 1, sizeof( NormEffect ) );
  CHECK_PTR(result);

  result->num_vars = 0;

  result->conditions = ( Fact * ) calloc( e->num_conditions, sizeof( Fact ) );
  result->num_conditions = e->num_conditions;
  for ( i = 0; i < e->num_conditions; i++ ) {
    result->conditions[i].predicate = e->conditions[i].predicate;
    for ( j = 0; j < garity[e->conditions[i].predicate]; j++ ) {
      result->conditions[i].args[j] = e->conditions[i].args[j];
    }
  }
  result->adds = ( Fact * ) calloc( e->num_adds, sizeof( Fact ) );
  result->num_adds = e->num_adds;
  for ( i = 0; i < e->num_adds; i++ ) {
    result->adds[i].predicate = e->adds[i].predicate;
    for ( j = 0; j < garity[e->adds[i].predicate]; j++ ) {
      result->adds[i].args[j] = e->adds[i].args[j];
    }
  }
  result->dels = ( Fact * ) calloc( e->num_dels, sizeof( Fact ) );
  result->num_dels = e->num_dels;
  for ( i = 0; i < e->num_dels; i++ ) {
    result->dels[i].predicate = e->dels[i].predicate;
    for ( j = 0; j < garity[e->dels[i].predicate]; j++ ) {
      result->dels[i].args[j] = e->dels[i].args[j];
    }
  }

  result->numeric_conditions_comp = ( Comparator * ) 
    calloc( e->num_numeric_conditions, sizeof( Comparator ) );
  result->numeric_conditions_lh = ( ExpNode_pointer * ) 
    calloc( e->num_numeric_conditions, sizeof( ExpNode_pointer ) );
  result->numeric_conditions_rh = ( ExpNode_pointer * ) 
    calloc( e->num_numeric_conditions, sizeof( ExpNode_pointer ) );
  for ( i = 0; i < e->num_numeric_conditions; i++ ) {
    result->numeric_conditions_comp[i] = e->numeric_conditions_comp[i];
    result->numeric_conditions_lh[i] = copy_Exp( e->numeric_conditions_lh[i] );
    result->numeric_conditions_rh[i] = copy_Exp( e->numeric_conditions_rh[i] );
  }
  result->num_numeric_conditions = e->num_numeric_conditions;
  result->numeric_effects_neft = ( NumericEffectType * ) 
    calloc( e->num_numeric_effects, sizeof( NumericEffectType ) );
  result->numeric_effects_fluent = ( Fluent * ) 
    calloc( e->num_numeric_effects, sizeof( Fluent ) );
  result->numeric_effects_rh = ( ExpNode_pointer * ) 
    calloc( e->num_numeric_effects, sizeof( ExpNode_pointer ) );
  for ( i = 0; i < e->num_numeric_effects; i++ ) {
    result->numeric_effects_neft[i] = e->numeric_effects_neft[i];
    result->numeric_effects_fluent[i].function = e->numeric_effects_fluent[i].function;
    for ( j = 0; j < gf_arity[e->numeric_effects_fluent[i].function]; j++ ) {
      result->numeric_effects_fluent[i].args[j] = e->numeric_effects_fluent[i].args[j];
    }
    result->numeric_effects_rh[i] = copy_Exp( e->numeric_effects_rh[i] );
  }
  result->num_numeric_effects = e->num_numeric_effects;

  result->next = NULL;
  result->prev = NULL;

  return result;

}



NormOperator *new_NormOperator( Operator *op )

{

  int i;

  NormOperator *result = ( NormOperator * ) calloc( 1, sizeof( NormOperator ) );
  CHECK_PTR(result);

  result->operator = op;

  result->num_vars = op->num_vars;
  for ( i = 0; i < op->num_vars; i++ ) {
    result->var_types[i] = op->var_types[i];
    result->inst_table[i] = -1;
  }
  result->num_removed_vars = 0;

  result->preconds = NULL;
  result->num_preconds = 0;

  result->numeric_preconds_comp = NULL;
  result->numeric_preconds_lh = NULL;
  result->numeric_preconds_rh = NULL;
  result->num_numeric_preconds = 0;

  result->effects = NULL;

  return result;

}




EasyTemplate *new_EasyTemplate( NormOperator *op )

{

  EasyTemplate *result = ( EasyTemplate * ) calloc( 1, sizeof( EasyTemplate ) );
  CHECK_PTR(result);

  result->op = op;

  result->prev = NULL;
  result->next = NULL;

  return result;

}



MixedOperator *new_MixedOperator( Operator *op )

{

  MixedOperator *result = ( MixedOperator * ) calloc( 1, sizeof( MixedOperator ) );
  CHECK_PTR(result);

  result->operator = op;

  result->preconds = NULL;
  result->num_preconds = 0;

  result->effects = NULL;

  return result;

}



PseudoActionEffect *new_PseudoActionEffect( void )

{

  PseudoActionEffect *result = 
    ( PseudoActionEffect * ) calloc( 1, sizeof( PseudoActionEffect ) );
  CHECK_PTR(result);

  result->conditions = NULL;
  result->num_conditions = 0;

  result->adds = NULL;
  result->num_adds = 0;
  result->dels = NULL;
  result->num_dels = 0;

  result->numeric_conditions_comp = NULL;
  result->numeric_conditions_lh = NULL;
  result->numeric_conditions_rh = NULL;
  result->num_numeric_conditions = 0;

  result->numeric_effects_neft = NULL;
  result->numeric_effects_fluent = NULL;
  result->numeric_effects_rh = NULL;
  result->num_numeric_effects = 0;

  result->next = NULL;

  return result;

}



PseudoAction *new_PseudoAction( MixedOperator *op )

{

  int i;

  PseudoAction *result = ( PseudoAction * ) calloc( 1, sizeof( PseudoAction ) );
  CHECK_PTR(result);

  result->operator = op->operator;
  for ( i = 0; i < op->operator->num_vars; i++ ) {
    result->inst_table[i] = op->inst_table[i];
  }

  result->preconds = op->preconds;
  result->num_preconds = op->num_preconds;

  result->numeric_preconds_comp = op->numeric_preconds_comp;
  result->numeric_preconds_lh = op->numeric_preconds_lh;
  result->numeric_preconds_rh = op->numeric_preconds_rh;
  result->num_numeric_preconds = op->num_numeric_preconds;

  result->effects = NULL;
  result->num_effects = 0;

  return result;

}



LnfExpNode *new_LnfExpNode( void )

{

  LnfExpNode *result = ( LnfExpNode * ) calloc( 1, sizeof( LnfExpNode ) );
  CHECK_PTR(result);

  result->num_pF = 0;
  result->num_nF = 0;

  result->c = 0;

  return result;

}



Action *new_Action( void )

{

  Action *result = ( Action * ) calloc( 1, sizeof( Action ) );
  CHECK_PTR(result);

  result->norm_operator = NULL;
  result->pseudo_action = NULL;

  result->next = NULL;

  return result;

}



void make_state( State *pointer, int ft, int fl ) 

{

  int i;

  pointer->F = ( int * ) calloc( ft, sizeof( int ) ); 
  pointer->f_D = ( Bool * ) calloc( fl, sizeof( Bool ) ); 
  pointer->f_V = ( float * ) calloc( fl, sizeof( float ) );

  for ( i = 0; i < fl; i++ ) {
    pointer->f_D[i] = FALSE;
  }

}



EhcNode *new_EhcNode( void )

{

  EhcNode *result = ( EhcNode * ) calloc( 1, sizeof( EhcNode ) );
  CHECK_PTR(result);

  make_state( &(result->S), gnum_ft_conn, gnum_fl_conn );

  result->father = NULL;
  result->next = NULL;

  return result;

}



EhcHashEntry *new_EhcHashEntry( void )

{

  EhcHashEntry *result = ( EhcHashEntry * ) calloc( 1, sizeof( EhcHashEntry ) );
  CHECK_PTR(result);

  result->ehc_node = NULL;

  result->next = NULL;

  return result;

}



PlanHashEntry *new_PlanHashEntry( void )

{

  PlanHashEntry *result = ( PlanHashEntry * ) calloc( 1, sizeof( PlanHashEntry ) );
  CHECK_PTR(result);

  result->next_step = NULL;

  result->next = NULL;

  return result;

}



BfsNode *new_BfsNode( void )

{

  BfsNode *result = ( BfsNode * ) calloc( 1, sizeof( BfsNode ) );
  CHECK_PTR(result);

  result->father = NULL;

  result->next = NULL;
  result->prev = NULL;

  return result;

}



BfsHashEntry *new_BfsHashEntry( void )

{

  BfsHashEntry *result = ( BfsHashEntry * ) calloc( 1, sizeof( BfsHashEntry ) );
  CHECK_PTR(result);

  result->bfs_node = NULL;

  result->next = NULL;

  return result;

}











/**********************
 * DELETION FUNCTIONS *
 **********************/












void free_TokenList( TokenList *source )

{

  if ( source ) {
    free_TokenList( source->next );
    if ( source->item ) {
      free( source->item );
    }
    free( source );
  }

}



void free_FactList( FactList *source )

{

  if ( source ) {
    free_FactList( source->next );
    free_TokenList( source->item );
    free( source );
  }

}



void free_ParseExpNode( ParseExpNode *n )

{

  if ( n ) {
    free_TokenList( n->atom );
    free_ParseExpNode( n->leftson );
    free_ParseExpNode( n->rightson );
    free( n );
  }

}



void free_PlNode( PlNode *node )

{
  
  if ( node ) {
    free_ParseExpNode( node->lh );
    free_ParseExpNode( node->rh );
    free_PlNode( node->sons );
    free_PlNode( node->next );
    free_TokenList( node->atom );
    free( node );
  }

}



void free_PlOperator( PlOperator *o )

{

  if ( o ) {
    free_PlOperator( o->next );

    if ( o->name ) {
      free( o->name );
    }
    
    free_FactList( o->params );
    free_PlNode( o->preconds );
    free_PlNode( o->effects );

    free( o );
  }

}



void free_Operator( Operator *o )

{

  if ( o ) {
    /* need not free more: the only point where that happens
     * is only directly after first allocation
     */

    if ( o->name ) {
      free( o->name );
    }

    free( o );
  } 

}



void free_ExpNode( ExpNode *n )

{

  if ( n ) {
    if ( n->fluent ) free( n->fluent );
    free_ExpNode( n->son );
    free_ExpNode( n->leftson );
    free_ExpNode( n->rightson );
    free( n );
  }

}



void free_WffNode( WffNode *w )

{

  if ( w ) {
    free_WffNode( w->son );
    free_WffNode( w->sons );
    free_WffNode( w->next );
    if ( w->var_name ) {
      free( w->var_name );
    }
    if ( w->fact ) free( w->fact );
    free_ExpNode( w->lh );
    free_ExpNode( w->rh );
    free( w );
  }

}



void free_NormEffect( NormEffect *e )

{

  int i;

  if ( e ) {
    free_NormEffect( e->next );

    if ( e->conditions ) {
      free( e->conditions );
    }
    if ( e->adds ) {
      free( e->adds );
    }
    if ( e->dels ) {
      free( e->dels );
    }

    if ( e->numeric_conditions_comp ) {
      free( e->numeric_conditions_comp );
    }
    for ( i = 0; i < e->num_numeric_conditions; i++ ) {
      free_ExpNode( e->numeric_conditions_lh[i] );
      free_ExpNode( e->numeric_conditions_rh[i] );
    }
    if ( e->numeric_conditions_lh ) {
      free( e->numeric_conditions_lh );
    }
    if ( e->numeric_conditions_rh ) {
      free( e->numeric_conditions_rh );
    }

    if ( e->numeric_effects_neft ) {
      free( e->numeric_effects_neft );
    }
    if ( e->numeric_effects_fluent ) {
      free( e->numeric_effects_fluent );
    }
    for ( i = 0; i < e->num_numeric_effects; i++ ) {
      free_ExpNode( e->numeric_effects_rh[i] );
    }
    if ( e->numeric_effects_rh ) {
      free( e->numeric_effects_rh );
    }

    free( e );
  }

}



void free_partial_Effect( Effect *e )

{

  if ( e ) {
    free_partial_Effect( e->next );

    free_WffNode( e->conditions );

    free( e );
  }

}



void free_NormOperator( NormOperator *o )

{

  int i;

  if ( o ) {

    if ( o->preconds ) {
      free( o->preconds );
    }
    if ( o->numeric_preconds_comp ) {
      free( o->numeric_preconds_comp );
    }
    for ( i = 0; i < o->num_numeric_preconds; i++ ) {
      free_ExpNode( o->numeric_preconds_lh[i] );
      free_ExpNode( o->numeric_preconds_rh[i] );
    }
    if ( o->numeric_preconds_lh ) {
      free( o->numeric_preconds_lh );
    }
    if ( o->numeric_preconds_rh ) {
      free( o->numeric_preconds_rh );
    }
    free_NormEffect( o->effects );

    free( o );
  }

}



void free_single_NormEffect( NormEffect *e )

{

  int i;

  if ( e ) {
    if ( e->conditions ) {
      free( e->conditions );
    }
    if ( e->adds ) {
      free( e->adds );
    }
    if ( e->dels ) {
      free( e->dels );
    }

    if ( e->numeric_conditions_comp ) {
      free( e->numeric_conditions_comp );
    }
    for ( i = 0; i < e->num_numeric_conditions; i++ ) {
      free_ExpNode( e->numeric_conditions_lh[i] );
      free_ExpNode( e->numeric_conditions_rh[i] );
    }
    if ( e->numeric_conditions_lh ) {
      free( e->numeric_conditions_lh );
    }
    if ( e->numeric_conditions_rh ) {
      free( e->numeric_conditions_rh );
    }

    if ( e->numeric_effects_neft ) {
      free( e->numeric_effects_neft );
    }
    if ( e->numeric_effects_fluent ) {
      free( e->numeric_effects_fluent );
    }
    for ( i = 0; i < e->num_numeric_effects; i++ ) {
      free_ExpNode( e->numeric_effects_rh[i] );
    }
    if ( e->numeric_effects_rh ) {
      free( e->numeric_effects_rh );
    }

    free( e );
  }

}



void free_single_EasyTemplate( EasyTemplate *t )

{

  if ( t ) {
    free( t );
  }

}



void free_TypedList( TypedList *t )

{

  if ( t ) {
    if ( t->name ) {
      free( t->name );
      t->name = NULL;
    }
    if ( t->type ) {
      free_TokenList( t->type );
      t->type = NULL;
    }
    free_TypedList( t->next );

    free( t );
  }

}



void free_TypedListList( TypedListList *t )

{

  if ( t ) {
    if ( t->predicate ) {
      free( t->predicate );
      t->predicate = NULL;
    }
    if ( t->args ) {
      free_TypedList( t->args );
      t->args = NULL;
    }
    free_TypedListList( t->next );

    free( t );
  }

}



void free_BfsNode( BfsNode *n )

{

  if ( n ) {
    free_BfsNode( n->next );
    free( n );
  }

}



void free_BfsHashEntry( BfsHashEntry *n )

{

  if ( n ) {
    free_BfsHashEntry( n->next );
    free( n );
  }

}
