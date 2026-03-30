
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
 * File: memory.h
 * Description: Creation / Deletion functions for all data structures.
 *
 * Author: Joerg Hoffmann / Frank Rittinger
 *
 *********************************************************************/ 






#ifndef _MEMORY_H
#define _MEMORY_H





char *new_Token( int len );
TokenList *new_TokenList( void );
FactList *new_FactList( void );
TypedList *new_TypedList( void );
TypedListList *new_TypedListList( void );
ParseExpNode *new_ParseExpNode( ExpConnective c );
PlNode *new_PlNode( Connective c );
PlOperator *new_PlOperator( char *name );
PlOperator *new_axiom_op_list( void );



Fact *new_Fact( void );
Fluent *new_Fluent( void );
FluentValue *new_FluentValue( void );
Facts *new_Facts( void );
FluentValues *new_FluentValues( void );
ExpNode *new_ExpNode( ExpConnective c );
WffNode *new_WffNode( Connective c );
Literal *new_Literal( void );
NumericEffect *new_NumericEffect( void );
Effect *new_Effect( void );
Operator *new_Operator( char *name, int norp );
NormEffect *new_NormEffect1( Effect *e );
NormEffect *new_NormEffect2( NormEffect *e );
NormOperator *new_NormOperator( Operator *op );
EasyTemplate *new_EasyTemplate( NormOperator *op );
MixedOperator *new_MixedOperator( Operator *op );
PseudoActionEffect *new_PseudoActionEffect( void );
PseudoAction *new_PseudoAction( MixedOperator *op );
LnfExpNode *new_LnfExpNode( void );
Action *new_Action( void );
void make_state( State *pointer, int ft, int fl );
EhcNode *new_EhcNode( void );
EhcHashEntry *new_EhcHashEntry( void );
PlanHashEntry *new_PlanHashEntry( void );
BfsNode *new_BfsNode( void );
BfsHashEntry *new_BfsHashEntry( void );







void free_TokenList( TokenList *source );
void free_FactList( FactList *source );
void free_ParseExpNode( ParseExpNode *n );
void free_PlNode( PlNode *node );
void free_PlOperator( PlOperator *o );
void free_Operator( Operator *o );
void free_ExpNode( ExpNode *n );
void free_WffNode( WffNode *w );
void free_NormEffect( NormEffect *e );
void free_partial_Effect( Effect *e );
void free_NormOperator( NormOperator *o );
void free_single_NormEffect( NormEffect *e );
void free_single_EasyTemplate( EasyTemplate *t );
void free_TypedList( TypedList *t );
void free_TypedListList( TypedListList *t );
void free_ActionEffect( ActionEffect *e );
void free_BfsNode( BfsNode *n );
void free_BfsHashEntry( BfsHashEntry *n );






#endif /* _MEMORY_H */
