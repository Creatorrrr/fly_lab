/** FLY LAB B v2. C may add signal types; never treat every numeric value as activation. */
export type NeuronId = string;
export interface RequestEnvelope { protocol:'flylab.protocol.v2'; requestId:number; op:string; payload:Record<string,unknown>; }
export type ResponseEnvelope<T> = {protocol:'flylab.protocol.v2';requestId:number;ok:true;result:T} | {protocol:'flylab.protocol.v2';requestId:number;ok:false;error:string};
export interface SignalDescriptor {kind:'activation'|'voltage'|'rate'|'spike'|string;unit:string;range?:[number,number]}
export interface CatalogEntry {id:NeuronId;name:string;type:string;origin:'anatomical'|'model';signal:SignalDescriptor}
/** Requested motor drive; negative forwardSpeed is an engineered contact backoff. */
export interface MotorCommand {yawRate:number;forwardSpeed:number;verticalSpeed:0}
export interface PhysicalTelemetry {
 backend:string;testDouble:boolean;physicalDt:number;controlDt:number;physicsTime:number;settlingTime:number;
 jointNames:string[];jointAngles:number[];jointTargets:number[];jointVelocities:number[];
 actuatorForces:number[];torqueUnit:string;contactsBW:number[];contactVectorsBW:number[][];
 adhesion:number[];cpgPhases:number[];descending:[number,number];fault:string|null;bodyModelHash:string;
}
export interface Frame {
 schema:'flylab.frame.v2';tick:number;simTime:number;sensorTick:number;
 neural:{ids:NeuronId[];values:number[];kind:string;unit:string;output:MotorCommand;avoidanceActive:boolean;recoveryActive:boolean;[key:string]:unknown};
 physics:PhysicalTelemetry;
 body:{position:[number,number,number];yaw:number;legs:Record<string,number[][]>;[key:string]:unknown};
 config:Record<string,unknown>;world:Record<string,unknown>;sensors:Record<string,unknown>;
}
