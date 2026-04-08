import { InstancedBufferAttribute, Matrix4, type Object3D } from 'three'
import {
  instancedBufferAttribute,
  nodeImmutable,
  positionPrevious,
  positionView,
  sub,
  uniform,
  vec4
} from 'three/tsl'
import {
  NodeUpdateType,
  TempNode,
  type NodeBuilder,
  type NodeFrame
} from 'three/webgpu'

import type { Node } from './node'

export class HighpVelocityNode extends TempNode {
  static override get type(): string {
    return 'HighpVelocityNode'
  }

  projectionMatrix?: Matrix4 | null

  private readonly currentProjectionMatrix = uniform('mat4')
  private readonly previousProjectionMatrix = uniform('mat4')

  private readonly currentModelViewMatrix = uniform('mat4')
  private readonly previousModelViewMatrix = uniform('mat4')
  private readonly objectModelViewMatrices = new WeakMap<Object3D, Matrix4>()

  constructor() {
    super('vec3')

    // Sequence:
    // - updateBefore() for the first object
    // - update() for the current frame
    // - updateAfter() for the first object
    // - updateBefore() for the next object
    // - updateAfter() for the next object
    // - ...
    this.updateType = NodeUpdateType.FRAME
    this.updateBeforeType = NodeUpdateType.OBJECT
    this.updateAfterType = NodeUpdateType.OBJECT
  }

  setProjectionMatrix(value: Matrix4 | null): this {
    this.projectionMatrix = value
    return this
  }

  private static getPositionAttribute(
    object: Object3D
  ): InstancedBufferAttribute | undefined {
    const material = (object as { material?: Record<string, unknown> }).material
    const fromUserData = (
      material?.userData as Record<string, unknown> | undefined
    )?.highpVelocityPositionAttribute
    if (
      fromUserData instanceof InstancedBufferAttribute &&
      fromUserData.itemSize >= 3
    ) {
      return fromUserData
    }

    const positionNode = material?.positionNode as
      | { attribute?: unknown }
      | undefined
    const attribute = positionNode?.attribute
    if (
      attribute instanceof InstancedBufferAttribute &&
      attribute.itemSize >= 3
    ) {
      return attribute
    }

    return undefined
  }

  private static getPreviousPositionNode(
    object: Object3D
  ): Node<'vec3'> | undefined {
    const material = (object as { material?: Record<string, unknown> }).material
    const previousPositionNode = (
      material?.userData as Record<string, unknown> | undefined
    )?.highpVelocityPreviousPositionNode
    if (previousPositionNode != null) {
      return previousPositionNode as Node<'vec3'>
    }
    return undefined
  }

  // Executed once per frame:
  override update({ camera }: NodeFrame): void {
    if (camera == null) {
      return
    }
    const {
      currentProjectionMatrix: current,
      previousProjectionMatrix: previous
    } = this

    const projectionMatrix = this.projectionMatrix ?? camera.projectionMatrix
    if (previous.value == null) {
      previous.value = new Matrix4().copy(projectionMatrix)
    } else {
      previous.value.copy(current.value)
    }
    current.value.copy(projectionMatrix)
  }

  // Executed once per object before rendering:
  override updateBefore({ object, camera }: NodeFrame): void {
    if (object == null || camera == null) {
      return
    }
    const {
      currentModelViewMatrix: current,
      previousModelViewMatrix: previous,
      objectModelViewMatrices: matrices
    } = this

    current.value.multiplyMatrices(
      camera.matrixWorldInverse,
      object.matrixWorld
    )
    previous.value = matrices.get(object) ?? current.value

  }

  // Executed once per object after rendering:
  override updateAfter({ object }: NodeFrame): void {
    if (object == null) {
      return
    }
    const {
      currentModelViewMatrix: current,
      objectModelViewMatrices: matrices
    } = this

    let matrix = matrices.get(object)
    if (matrix == null) {
      matrix = new Matrix4()
      matrices.set(object, matrix)
    }
    matrix.copy(current.value)

  }

  override setup(builder: NodeBuilder): unknown {
    const object = (builder as { object?: Object3D }).object
    const previousPositionNode =
      object != null
        ? HighpVelocityNode.getPreviousPositionNode(object)
        : undefined
    const positionAttribute =
      previousPositionNode == null && object != null
        ? HighpVelocityNode.getPositionAttribute(object)
        : undefined

    const currentClip = this.currentProjectionMatrix
      .mul(vec4(positionView, 1))
      .toVertexStage()
    const previousClip = (
      previousPositionNode == null
        ? positionAttribute == null
          ? this.previousProjectionMatrix
              .mul(this.previousModelViewMatrix)
              .mul(positionPrevious)
          : this.previousProjectionMatrix
              .mul(this.previousModelViewMatrix)
              .mul(vec4(instancedBufferAttribute(positionAttribute, 'vec3'), 1))
        : this.previousProjectionMatrix
            .mul(this.previousModelViewMatrix)
            .mul(vec4(previousPositionNode, 1))
    ).toVertexStage()

    // Perspective divisions cannot be performed in the vertex shader.
    // See: http://john-chapman-graphics.blogspot.com/2013/01/per-object-motion-blur.html
    const currentNDC = currentClip.xyz.div(currentClip.w)
    const previousNDC = previousClip.xyz.div(previousClip.w)

    return sub(currentNDC, previousNDC)
  }
}

export const highpVelocity = /*#__PURE__*/ nodeImmutable(HighpVelocityNode)
